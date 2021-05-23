import torch 
import numpy as np 
import torchvision 
from config import device 

def bbox_wh2xy(box1):
    box2 = torch.zeros(box1.shape).to(device)
    x,y,w,h = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    box2[:,0] = x
    box2[:,1] = y
    box2[:,2] = x + w
    box2[:,3] = y + h 
    return box2

#批交并比
def bbox_batch_iou(box1, box2):
    #将xywh格式的bbox转化为xyxy两点式

    box1 = bbox_wh2xy(box1)
    box2 = bbox_wh2xy(box2)

    #使用两点式计算iou
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.max(inter_rect_x2 - inter_rect_x1, torch.zeros(inter_rect_x2.shape).to(device)) * torch.max(
        inter_rect_y2 - inter_rect_y1, torch.zeros(inter_rect_x2.shape).to(device))

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

#联合交并比，转化为xyxyx格式
def bbox_union_iou(bbox_a, bbox_b):
    bbox_a = bbox_wh2xy(bbox_a)
    bbox_b = bbox_wh2xy(bbox_b)

    #利用广播机制计算交并比
    tl = torch.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = torch.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = torch.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

#nms 使用 keep = torchvision.ops.nms(boxes, scores, iou_threshold)

#输入xywh格式的box，输出loc
def bbox2loc(src, dst):
    
    x,y,w,h = src[:,0], src[:,1], src[:,2], src[:,3]
    bx,by,bw,bh = dst[:,0], dst[:,1], dst[:,2], dst[:,3]

    eps = torch.finfo(h.dtype).eps
    h[h<eps] = eps 
    w[w<eps] = eps 
    
    dx = (bx-x) / w
    dy = (by-y) / h
    dw = torch.log(bw / w)
    dh = torch.log(bh / h)
    
    loc = torch.stack((dx,dy,dw,dh),1)
    return loc

def loc2bbox(src, loc):
    dst = torch.zeros(src.shape).to(device)
    dx,dy,dw,dh = loc[:,0], loc[:,1], loc[:,2], loc[:,3]
    x,y,w,h = src[:,0], src[:,1], src[:,2], src[:,3]
   
    bx = dx*w + x 
    by = dy*h + y 
    bw = torch.exp(dw) * w 
    bh = torch.exp(dh) * h

    dst[:,0], dst[:,1], dst[:,2], dst[:,3] = bx, by, bw, bh
    return dst 
    
if __name__ == '__main__':
    a = torch.Tensor([[1,1,2,2],[1,1,1,1],[1,1,1,1]]).to(device) 
    b = torch.Tensor([[1,4,4,4],[1,3,3,1],[0.8,0.1,0.1,0.1]]).to(device) 
    #iou = bbox_iou(a,b)
    #union_iou = bbox_union_iou(a,b)
    loc = bbox2loc(a,b)
    c = loc2bbox(a,loc)
    