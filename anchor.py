import torch 
import numpy as np 
from data import myCityscapes
from torch.utils.data import DataLoader
from torchvision import transforms
from data import jsonTransform
import matplotlib.pyplot as plt
from data import show_bboxes_img
from data import show_bboxes_compare_img
from config import device
from bbox import bbox_union_iou
from torchvision.ops import nms 
from bbox import bbox_wh2xy
from bbox import bbox2loc

def gererate_base_anchor(ratios=[4,6,8],scales=[0.1,0.06,0.02]):
    anchor = torch.Tensor(len(ratios)*len(scales),4).to(device)
    for i,r in enumerate(ratios):
        for j,s in enumerate(scales):
            w = s 
            h = r * w
            x0 = -w/2
            y0 = -h/2 
            anchor[i*len(ratios)+j,:] = torch.Tensor([x0,y0,w,h]).to(device)
    #[9,4]
    return anchor

def gererate_all_anchor(num_gridx=32,num_gridy=16):
    
    x = (torch.arange(0,num_gridx).to(device) + 0.5) / num_gridx  
    y = (torch.arange(0,num_gridy).to(device) + 0.5) / num_gridy
   
    x,y = torch.meshgrid(x,y)
    x = x.reshape(-1)
    y = y.reshape(-1)
    w = torch.zeros(x.shape).to(device)
    h = torch.zeros(x.shape).to(device)

    shift = torch.stack([x,y,w,h],axis=1)
    shift = shift.unsqueeze(0).permute(1,0,2)

    anchor = gererate_base_anchor()
    anchor = anchor.unsqueeze(0)

    anchors = anchor + shift 
    anchors = anchors.reshape(-1,4)
    return anchors

def create_anchor_target(anchor,bbox, pos_iou_thresh=0.7, neg_iou_thresh=0.3, n_sample=128):
    ious = bbox_union_iou(anchor,bbox)
    max_ious, argmax_ious = ious.max(axis=1) 
    gt_argmax_ious = ious.argmax(axis=0) 
    
    #产生每个anchor位移到最接近的bbox之间的loc用于训练
    loc = bbox2loc(anchor, bbox[argmax_ious])
   
    #产生label用于训练fpn score，使用正负采样的方式
    label = torch.zeros(len(anchor)).to(device) - 1 
    label[max_ious < neg_iou_thresh] = 0
    #label[gt_argmax_ious] = 1
    label[max_ious >= pos_iou_thresh] = 1

    #负采样
    n_pos = torch.sum(label==1).item()
    n_neg = n_pos

    neg_index = torch.where(label == 0)[0]
    if len(neg_index) > n_neg:
        disable_index = np.random.choice(neg_index.cpu(), size=(len(neg_index) - n_neg), replace=False)
        label[disable_index] = -1

    return loc,label 

if __name__ == '__main__':
    #anchor = gererate_base_anchor()
    anchors = gererate_all_anchor()

    myTransform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    BATCH = 4
    dataset = myCityscapes('./datasets/', split='train', mode='person', target_type='person',transform=myTransform,target_transform=jsonTransform)
    dataloader = DataLoader(dataset, batch_size=BATCH,shuffle=True)
    for i,data in enumerate(dataloader):
        if i > 0:
            break
        img,target = data
        bboxes,clses = target 
        img = img[0]
        bboxes = bboxes[0]
        clses = clses[0]
        bboxes = bboxes[clses==1]
        if len(bboxes) == 0:
            bboxes = torch.zeros(1,4).to(device)

        max_ious, argmax_ious, labels = create_anchor_target(anchors,bboxes)
        pos_anchor = anchors[labels==1]
        scores = max_ious[labels==1]

        keep = nms(bbox_wh2xy(pos_anchor), scores, 0.5)
        pos_anchor = pos_anchor[keep]
        
        #show_bboxes_img(img, pos_anchor,'dump/nms_anchor.png')
        show_bboxes_compare_img(img, pos_anchor, bboxes, 'dump/res_anchor.png')
