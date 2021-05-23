import torch 
import numpy as  np 
if __name__ == '__main__':
    a = torch.Tensor([[1,2,3,4],[5,6,7,8]])
    b = a.view(-1)
    print(b)
    b = torch.Tensor([1,0,0,0,1])
