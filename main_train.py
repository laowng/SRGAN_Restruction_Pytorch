import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from net import Net,DisNet
from dataset import TrainDatasetFromFolder
from draw import draw
from matplotlib import pyplot as plt
from loss import GLoss ,DLoss

#全局参数设置
parser=argparse.ArgumentParser(description="VDSR,全局参数")
parser.add_argument("--batch_size","-b",  type=int, default=50, help="batch_size set,默认：128")
parser.add_argument("--nEpochs",   "-e",  type=int, default=100, help="总训练次数，默认：50")
parser.add_argument("--gen_lr",    "-gl", type=float, default=0.0001, help="learning rate, 默认：0.1")
parser.add_argument("--dis_lr",    "-dl", type=float, default=0.0001, help="learning rate, 默认：0.1")
parser.add_argument("--step",      "-s",  type=int, default=20, help="步长，默认：20")
parser.add_argument("--cuda",      "-c",  action="store_true", help="是否使用cuda，默认：false")
parser.add_argument("--resume",    "-r",  default="", help="继续上次训练的路径，默认不使用")
parser.add_argument("--start_epoch",'-se', type=int,default=1,help="开始的epoch，默认：1")
parser.add_argument("--clip",      '-cl', type=float, default=0.9,help="梯度出现的最大值,防止梯度爆炸")
parser.add_argument("--momentum",  "-m", default=0.9, type=float, help="动量，默认: 0.9")
parser.add_argument("--weight_decay", "-wd", default=0.0001, type=float, help="权重衰减，默认: 1e-4")

def main():
    global par
    par = parser.parse_args()
    print(par)

    print("===> 建立模型")
    # Gmodel=Net() #模型
    Dmodel = DisNet(96)
    Gmodel=torch.load("./checkpoint/pre_model.pth")["model"]
    # Dmodel = torch.load("./checkpoint/pre_dmodel.pth")["model"]

    GCriterion = GLoss()
    # GCriterion = nn.MSELoss()
    DCriterion = DLoss


    print("===> 加载数据集")
    train_set = TrainDatasetFromFolder("./data/train_set.h5")
    train_loader = DataLoader(dataset=train_set,num_workers=1,batch_size=par.batch_size, shuffle=True)


    print("===> 设置 GPU")
    cuda = par.cuda
    if cuda :
        if torch.cuda.is_available():
            Gmodel.cuda()
            Dmodel.cuda()
            GCriterion.cuda()
        else:raise Exception("没有可用的显卡设备")

    # optionally resume from a checkpoint
    if par.resume:
        if os.path.isfile(par.resume):
            checkpoint=torch.load(par.resume)
            par.start_epoch=checkpoint['epoch']
            Gmodel.load_state_dict(checkpoint["model"].statedict())

    print("===> 设置 优化器")
    Goptimizer = optim.Adam(Gmodel.parameters(), lr=par.gen_lr)
    Doptimizer = optim.Adam(Dmodel.parameters(), lr=par.dis_lr)
    print("===> 进行训练")
    # plt.figure(figsize=(8, 6), dpi=80)
    draw_list = []
    for epoch in range(par.start_epoch, par.nEpochs + 1):
        draw_list=train(train_loader, Goptimizer, Doptimizer, Gmodel,Dmodel, GCriterion, DCriterion, epoch, draw_list)
        save_checkpoint(Gmodel, epoch)
        save_dcheckpoint(Dmodel,epoch)
        # draw(range(1,len(draw_list)+1), 10, draw_list, 10,{"EPOCH:":epoch,"LR:":round(Goptimizer.param_groups[0]["lr"],4)})
    # plt.show()





def adjust_learning_rate(optimizer, epoch,step,end_step, rate):# 学习率修改函数
    if epoch > end_step:
        return
    if epoch == 1 or epoch % step-1 != 0:
        return
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * rate


def train(training_data_loader, Goptimizer,Doptimizer, Gmodel, Dmodel, GCriterion,DCriterion,epoch,draw_list=[]):
    # adjust_learning_rate(Goptimizer, epoch, par.step,600,0.1)
    # adjust_learning_rate(Doptimizer, epoch, par.step, 70, 0.1)
    print("Epoch = {}, Glr = {} Dlr = {}".format(epoch, Goptimizer.param_groups[0]["lr"],Doptimizer.param_groups[0]["lr"]))

    Gloss_sum=0
    Dloss_sum=0

    Dmodel.train()
    Gmodel.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0],batch[1]
        if par.cuda:
            input = input.cuda()
            target = target.cuda()
        Gout = Gmodel(input)
        Dout1=Dmodel(target)
        # Dout2=Dmodel(Gout.detach())
        Dout2=Dout3=Dmodel(Gout)
        #
        Dloss = DCriterion(Dout1,Dout2)
        Gloss = GCriterion(Dout3,Gout,target)
        # Gloss = GCriterion(Gout,target)

        Doptimizer.zero_grad()
        Dloss.backward(retain_graph=True)
        Doptimizer.step()

        Goptimizer.zero_grad()
        Gloss.backward()
        Goptimizer.step()

        Gloss_sum+=Gloss.item()
        Dloss_sum+=Dloss.item()
        print("iteration:",iteration)
    print("===> Epoch[{}](iterations:{}) DLoss: {:.10f} , GLoss: {:.10f}".format(epoch, len(training_data_loader), Dloss_sum,Gloss_sum))
    jilu = open("./jilu.txt", "a")
    jilu.writelines("\n" + str(epoch) + ":  Dlr:" + str(Doptimizer.param_groups[0]["lr"])[0:7] + "     " + str(Dloss_sum) +"Glr:" + str(Goptimizer.param_groups[0]["lr"])[0:7] + "     " + str(Gloss_sum))
    jilu.close()
    draw_list.append(Dloss_sum)
    return draw_list

def save_checkpoint(model, epoch):
    if epoch % 1 == 0:
        model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
        state = {"epoch": epoch ,"model": model}
        if not os.path.exists("checkpoint/"):
            os.makedirs("checkpoint/")

        torch.save(state, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))

def save_dcheckpoint(model, epoch):
    if epoch % 1 == 0:
        model_out_path = "checkpoint/" + "dmodel_epoch_{}.pth".format(epoch)
        state = {"epoch": epoch ,"model": model}
        if not os.path.exists("checkpoint/"):
            os.makedirs("checkpoint/")

        torch.save(state, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()