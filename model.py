from torch import nn
import torch 
from torchvision.models import vgg16
import torch.nn.functional as F
from anchor import gererate_all_anchor 
from config import device 
from anchor import create_anchor_target 
from torchvision.ops import nms 
from bbox import loc2bbox
from data import show_bboxes_compare_img 
from torch.utils.data import DataLoader
from torchvision import transforms
from data import jsonTransform 
from data import myCityscapes 
import tqdm 
from torchvision.ops import nms
from bbox import bbox_wh2xy 
from config import * 
from torchsummary import summary

def get_vgg16_extracter():
    model = vgg16(pretrained=True).to(device)
    features = list(model.features)[:30]

    #冻结某些层
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    features = nn.Sequential(*features)
    #用vgg16提取特征返回16倍下采样后的结果
    return features

class FPN(nn.Module):
    #FPN,输出每个锚点的归一化之后的loc和pred，
    def __init__(self,n_anchors=9,in_channel=512,out_channel=512):
        super().__init__()
        self.extracter = get_vgg16_extracter()
        #self.extracter = CONV5()
        self.conv = nn.Sequential(nn.Conv2d(in_channel,out_channel,3,1,1),nn.BatchNorm2d(out_channel), nn.ReLU())
        self.loc = nn.Conv2d(out_channel, n_anchors*4 ,3, 1, 1)
        self.score = nn.Conv2d(out_channel,n_anchors, 3, 1,1)

    def forward(self,x):
        n_batch = x.shape[0]
        x = self.extracter(x)
        x = self.conv(x)

        pred_score = self.score(x)
        pred_loc = self.loc(x)
        
        pred_score = pred_score.permute(0,2,3,1).contiguous()
        pred_loc = pred_loc.permute(0,2,3,1).contiguous()
        pred_score = pred_score.view(n_batch,-1)
        
        pred_loc = pred_loc.view(n_batch,-1,4)
        pred_score = torch.sigmoid(pred_score)

        return pred_loc, pred_score

class FPNTrainer():
    def __init__(self,img_size):
        self.epoches = 30
        self.lr = 1e-2

        w,h = img_size
        self.anchors = gererate_all_anchor(num_gridx=w//16,num_gridy=h//16)
        self.model = FPN().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.lr)
        self.box_loss_weight = 1
        self.pred_loss_weight = 1

    def train(self, dataloader, validloader):
        for e in range(self.epoches):
            tot_box_loss = 0
            tot_pred_loss = 0
            tot_num = 0
            for i,data in enumerate(tqdm.tqdm(dataloader)):
                #选取部分数据集训练
                if i > 100:
                    break
                else:
                    tot_num += 1
                img,target = data 
                bboxes,clses = target 
                img = img.to(device)
                bboxes = bboxes.to(device)
                clses = clses.to(device)
                pred_loc, pred_score = self.model(img)
                
                #随机选取一个样本
                img = img[0]
                bboxes = bboxes[0]
                clses = clses[0]
                pred_loc = pred_loc[0]
                pred_score = pred_score[0]
                
                #根据cls作为掩码获取bbox，没有bbox的不进行学习，利用bbox预先生成每个锚点需要学习的偏移量，并进行采样
                bboxes = bboxes[clses==1]
                if len(bboxes) == 0:
                    continue
                true_loc, labels = create_anchor_target(self.anchors, bboxes)
                if torch.sum(labels==1) == 0:
                    continue
                #取出loc中正样本的部分用smoothF1计算损失，score中正负样本的部分用MSEloss计算损失
                
                pred_loc = pred_loc[labels==1]
                true_loc = true_loc[labels==1]

                pos_pred_score = pred_score[labels==1]
                pos_true_score = labels[labels==1]
                neg_pred_score = pred_score[labels==0]
                neg_true_score = labels[labels==0]
                
                box_loss = self.box_loss_weight * F.smooth_l1_loss(pred_loc,true_loc)
                pred_loss = self.pred_loss_weight * (F.mse_loss(pos_pred_score, pos_true_score) + F.mse_loss(neg_pred_score,neg_true_score))
                
                loss = box_loss + pred_loss 
                tot_box_loss += box_loss.item()
                tot_pred_loss += pred_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            tot_box_loss /= tot_num
            tot_pred_loss /= tot_num
            print("e:",e,"box_loss:",tot_box_loss,"pred_loss:",tot_pred_loss)
            savepath = 'dump/' + str(e) +'.png'
            self.test(dataloader,savepath)

    @torch.no_grad()
    def test(self, dataloader, savepath):
        for i, data in enumerate(dataloader):
            if i>0:
                break 
            img,target = data 
            bboxes,clses = target 
            img = img.to(device)
            bboxes = bboxes.to(device)
            clses = clses.to(device)
            pred_loc, pred_score = self.model(img)
            
            #随机选取一个样本
            img = img[0]
            bboxes = bboxes[0]
            clses = clses[0]
            pred_loc = pred_loc[0]
            pred_score = pred_score[0]
            
            #根据cls作为掩码获取bbox，没有bbox的不进行学习，利用bbox预先生成每个锚点需要学习的偏移量，并进行采样
            bboxes = bboxes[clses==1]
            if len(bboxes) == 0:
                continue
            
            #边框回归与置信度分类同时进行
            pred_bbox = loc2bbox(self.anchors, pred_loc)
            pred_bbox = pred_bbox[pred_score>0.5]
            pred_score = pred_score[pred_score>0.5]
            keep = nms(bbox_wh2xy(pred_bbox),pred_score,0.4)
            show_bboxes_compare_img(img,pred_bbox,bboxes,savepath)

            #只使用边框回归
            """
            pred_bbox = loc2bbox(self.anchors, pred_loc)
            pred_bbox = pred_bbox[labels==1]
            labels = labels[labels==1]
            show_bboxes_compare_img(img,pred_bbox,bboxes,savepath)
            """

            #只使用置信度分类
            """
            true_loc, labels = create_anchor_target(self.anchors, bboxes)
            true_box = loc2bbox(self.anchors, true_loc)
            true_box = true_box[pred_score>0.5]
            pred_score = pred_score[pred_score>0.5]
            keep = nms(bbox_wh2xy(true_box),pred_score,0.4)
            true_box = true_box[keep]
            show_bboxes_compare_img(img,true_box,bboxes,savepath)
            """
            #pred_bbox = pred_bbox[keep]
            #show_bboxes_compare_img(img,pred_bbox,bboxes,savepath)

            


               
if __name__ == '__main__':
    
    myTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMG_SIZE[0]),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = myCityscapes('./datasets/', split='train', mode='person', target_type='person',transform=myTransform,target_transform=jsonTransform)
    dataloader = DataLoader(dataset, batch_size=BATCH,shuffle=True)
    validset = myCityscapes('./datasets/', split='val', mode='person', target_type='person',transform=myTransform,target_transform=jsonTransform)
    validloader = DataLoader(dataset, batch_size=BATCH,shuffle=True)
    
    fpn = FPNTrainer(IMG_SIZE)
    print(fpn.model)
    fpn.train(dataloader,validloader)