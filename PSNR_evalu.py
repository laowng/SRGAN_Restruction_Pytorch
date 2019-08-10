# 测试集测试
import psnr
import torch
import random
import torch.utils.data as Data
from torchvision.transforms import ToTensor, ToPILImage
from dataset import TrainDatasetFromFolder
from PIL import Image
import time

def h2ll(HRim,scale_factor):#仅仅下采样
    w,h=HRim.size
    LRim=HRim.resize((int(w//scale_factor),int(h//scale_factor)),Image.BICUBIC)
    return LRim


tran_im=ToPILImage()
tran_ten=ToTensor()


test_set = TrainDatasetFromFolder("./data/test_set.h5")#h5数据集制作工具在data中，可自己制作
test_loader = Data.DataLoader(dataset=test_set,num_workers=1,batch_size=40, shuffle=False)
net=torch.load("./checkpoint/pre_model.pth")["model"]

net.cpu()
i=random.randint(0,50)

aa,bb=test_loader.dataset[i]

lim1=tran_im(bb)
# label=tran_im(bb)
hh=h2ll(lim1,0.25)

# print("CUBIC_psnr:",psnr.psnr(h2ll(lim1,4),label))

T_X=torch.unsqueeze(tran_ten(lim1),0)


prediction=net(T_X)
prediction[prediction<0]=0
prediction[prediction>1]=1



lim2=tran_im(prediction[0])
lim2.show()

time.sleep(0.5)
# print("NET_psnr:",psnr.psnr(lim2,label))
hh.show()


