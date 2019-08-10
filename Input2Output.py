from PIL import Image
import os
import torch
from torchvision.transforms import ToTensor, ToPILImage


tran_im=ToPILImage()
tran_ten=ToTensor()

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.bmp'])


if __name__=="__main__":
    net = torch.load("./checkpoint/pre_model.pth")["model"]
    net.cpu()
    LR_dir = "./set5input"
    LRimage_filenames = [os.path.join(LR_dir, x) for x in os.listdir(LR_dir) if is_image_file(x)]
    LRimage_filenames.sort()
    scales = [2, 3, 4]
    for LRdir in LRimage_filenames:
        (filepath, tempfilename) = os.path.split(LRdir)
        (filename, extension) = os.path.splitext(tempfilename)
        filepath = "./set5output"
        LRim = Image.open(LRdir)
        LRim = tran_ten(LRim)
        LRim = torch.unsqueeze(LRim,0)
        HRim = net(LRim)
        HRim = HRim[0]
        HRim[HRim < 0] = 0
        HRim[HRim > 1] = 1
        HRim = tran_im(HRim)

        HRname = os.path.join(filepath,"HR_"+filename+extension)
        print(HRname)
        HRim.save(HRname)