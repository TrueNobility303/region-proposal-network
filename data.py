import json
import os
import torchvision
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from config import device,MAX_OB,IMG_SIZE

# from https://github.com/pytorch/vision/tree/master/torchvision
# REF: https://blog.csdn.net/weixin_41424926/article/details/105383064

#基于torchvision里的cityscape数据集，增加了person以支持行人目标检测任务
class myCityscapes(VisionDataset):

    #Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            mode: str = "fine",
            target_type: Union[List[str], str] = "instance",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        if mode == 'fine':
            self.mode = 'gtFine'
        elif mode == 'coarse':
            self.mode = 'gtCoarse'
        elif mode == 'person':
            self.mode = 'gtBbox'
        #self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        if mode == 'person':
            self.targets_dir = os.path.join(self.root, 'gtBboxCityPersons', split)
        else:
            self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []

        verify_str_arg(mode, "mode", ("fine", "coarse","person"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("instance", "semantic", "polygon", "color", "person"))
         for value in self.target_type]

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            if split == 'train_extra':
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainextra.zip'))
            else:
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainvaltest.zip'))

            if self.mode == 'gtFine':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))
            elif self.mode == 'gtCoarse':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '.zip'))
            elif self.mode == 'gtBbox':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_cityPersons_trainval.zip'))

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.mode, t))
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon' or t == 'person':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'person':
            return 'gtBboxCityPersons.json'

myTransform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#px,py 表示 gridx,gridy 到cx，cy的偏移
def box2subbox(bboxes,clses):
    subboxes = torch.zeros((NUMGRID,NUMGRID,NUMBBOX,4))
    subclses = torch.zeros((NUMGRID,NUMGRID,NUMBBOX))

    gridsize = 1.0 / NUMGRID

    for box,cl in zip(bboxes,clses):
        if cl == 0:
            continue
        x,y,w,h = box 
        cx = x + w/2
        cy = y + h/2

        gridx = int(cx/gridsize)
        gridy = int(cy/gridsize)
        px = cx/gridsize - gridx
        py = cy/gridsize - gridy

        for i in range(NUMBBOX):
            subboxes[gridx,gridy,i,:] = torch.Tensor([px,py,w,h]) 
            subclses[gridx,gridy,i] = 1     
    
    return subboxes,subclses

def jsonTransform(x):
    #将json格式读取入bbox，返回bbox和cls，bbox最多为MAX_OB个，bbox归一化至[0,1),clss为0/1
    obs = x['objects']
    boxes = torch.Tensor(np.zeros((MAX_OB,4))).to(device)
    clses = torch.Tensor(np.zeros(MAX_OB)).to(device)
    
    cnt = 0
    for ob in obs:
        if  ob['label'] == 'ignore':
            continue
        #舍弃太小的物体 
        eps = 1e-2
        if ob['bbox'][2] / 2048 < eps and  ob['bbox'][3] / 1024 < eps:
            continue
        else:
            boxes[cnt] = torch.Tensor(ob['bbox']).to(device)
            boxes[cnt,0] /= 2048
            boxes[cnt,1] /= 1024
            boxes[cnt,2] /= 2048
            boxes[cnt,3] /= 1024
            clses[cnt] = 1
            cnt += 1
            if cnt == MAX_OB:
                break 
        
    return boxes,clses

def show_bboxes_img(img, boxes, savepath):
    img = torch.clip(img,min=0,max=1)
    img = img.float().detach().cpu().permute(1,2,0).numpy()
    h,w,_ = img.shape
    boxes = boxes.detach().cpu().numpy()
    ax =plt.subplot(1,1,1)
    plt.imshow(img)
    
    for box in boxes:
        box[0] = box[0]*2048
        box[1] = box[1]*1024
        box[2] = box[2]*2048
        box[3] = box[3]*1024

        ret = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1, edgecolor='r',facecolor='none') 
        ax.add_patch(ret)

    plt.axis('off')
    plt.savefig(savepath)

def show_bboxes_compare_img(img,boxes,gtboxes,savepath):
    img = torch.clip(img,min=0,max=1)
    img = img.float().detach().cpu().permute(1,2,0).numpy()
    h,w,_ = img.shape
    boxes = boxes.detach().cpu().numpy()
    gtboxes = gtboxes.detach().cpu().numpy()

    plt.figure()
    plt.clf()
    ax =plt.subplot(2,1,1)
    plt.imshow(img)
    
    for box in boxes:
        box[0] = box[0]*w
        box[1] = box[1]*h
        box[2] = box[2]*w
        box[3] = box[3]*h

        ret = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1, edgecolor='r',facecolor='none') 
        ax.add_patch(ret)

    plt.axis('off')

    ax = plt.subplot(2,1,2)
    plt.imshow(img)
    for gtbox in gtboxes:

        gtbox[0] = gtbox[0]*w
        gtbox[1] = gtbox[1]*h
        gtbox[2] = gtbox[2]*w
        gtbox[3] = gtbox[3]*h
        
        ret = patches.Rectangle((gtbox[0],gtbox[1]),gtbox[2],gtbox[3],linewidth=1, edgecolor='b',facecolor='none') 
        ax.add_patch(ret)

    plt.axis('off')

    plt.savefig(savepath)
    plt.close()

if __name__ == '__main__':
    BATCH = 4
    dataset = myCityscapes('./datasets/', split='train', mode='person', target_type='person',transform=myTransform,target_transform=jsonTransform)
    dataloader = DataLoader(dataset, batch_size=BATCH,shuffle=True)
    for i,data in enumerate(dataloader):
        if i > 0:
            break
        img,target = data
        bboxes, clses = target 
        img = img[0]
        bboxes = bboxes[0]
        clses = clses[0]
        bboxes = bboxes[clses==1]
        show_bboxes_img(img, bboxes, 'dump/pic.png')






