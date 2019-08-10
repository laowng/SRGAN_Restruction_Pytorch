import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16

class GLoss(nn.Module):
    def __init__(self):
        super(GLoss,self).__init__()
        vgg=vgg16(pretrained=True)
        lossnet=vgg.features
        lossnet.eval()
        for param in lossnet.parameters():
            param.requires_grad=False
        self.lossnet=lossnet
        self.mseloss=nn.MSELoss()

    def forward(self,Dout,Gout,data):
        adversarial_loss = -torch.mean(torch.log(Dout+1e-10))
        perception_loss=self.mseloss(self.lossnet(data),self.lossnet(Gout))
        image_loss=self.mseloss(data,Gout)

        return 0.006*perception_loss+image_loss+0.001*adversarial_loss


def DLoss(Dout1,Dout2):
    return -torch.mean((torch.log(Dout1+1e-10)+torch.log((1-Dout2))))