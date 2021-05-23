#配置文件，定义全局变量
import torch 
device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
MAX_OB = 100
IMG_SIZE = (1024,2048)
BATCH = 1
