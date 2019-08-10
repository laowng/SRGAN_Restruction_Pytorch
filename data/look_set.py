from torchvision.transforms import ToPILImage,ToTensor
import h5py
import numpy as np
from PIL import Image
import time
import torch
tran1=ToPILImage()
tran2=ToTensor()
file=h5py.File("./train_set.h5")
data = file["train/data"]
label = file["train/label"]
print("data:",data)
print("label:",label)

for i in range(3):
    im1=torch.from_numpy(data[i])
    im2=torch.from_numpy(label[i])
    im1=tran1(torch.from_numpy(data[i]))
    im2=tran1(torch.from_numpy(label[i]))
    print(data)
    im1.show()
    time.sleep(1)
    im2.show()
    time.sleep(1)
file.close()