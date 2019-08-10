from torchvision.transforms import ToTensor
from PIL import Image
from os.path import join
from os import listdir
from HR_2_LR import h2l ,h2ll
import random
import h5py

def save_h5(pfile,data,label):
    data=data.reshape((1,*data.shape))
    if not pfile.__contains__(label):
        shapelist = list(data.shape)
        shapelist[0]=0
        dataset = pfile.create_dataset(label, data=data, maxshape=(None,*data.shape[1:4]),chunks=True)
        dataset.resize(data.shape)
        dataset[0:data.shape[0]]=data
        return
    else:
        dataset=pfile[label]

    shapelist = list(data.shape)
    len_old=dataset.shape[0]
    len_new=len_old+data.shape[0]
    shapelist[0]=len_new
    dataset.resize(tuple(shapelist))
    dataset[len_old:len_new]=data

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.bmp'])

def main():
    global pfile,tran1,HR_dir
    label_data="train/data"
    label_label = "train/label"
    num=100  #数据集数量  此数量由论文得到
    size=288    #数据集尺寸
    scale=4     #缩放因子
    HRimage_filenames = [join(HR_dir, x) for x in listdir(HR_dir) if is_image_file(x)]
    HRimage_filenames.sort()
    iter=0
    length=len(HRimage_filenames)
    while iter <num:
        if iter<700000:scale=4
        elif iter<1400000:scale=4
        else: scale=4   #4个尺寸
        for ipath in HRimage_filenames:
            if iter<num:
                hr_image=Image.open(ipath)
                # # 随机翻转
                # trans=rotation=random.randint(0,1)
                # hr_image=hr_image.transpose(trans)
                # #随机旋转
                # rotation=random.randint(0,15)
                # hr_image=hr_image.rotate(rotation)
                # 随机裁切
                b = size.__divmod__(2)
                w, h = hr_image.size
                x1=random.randint(b[0],w-b[0]-b[1])
                y1=random.randint(b[0],h-b[0]-b[1])
                hr_image = hr_image.crop((x1 - b[0], y1 - b[0], x1 + b[0] + b[1], y1 + b[0] + b[1]))
                #采样
                w, h = hr_image.size
                hr_image = hr_image.crop((0, 0, w, h))
                lr_image = h2ll(hr_image,scale)
                data_lr=tran1(lr_image)
                data_hr=tran1(hr_image)
                save_h5(pfile, data_lr, label_data)
                save_h5(pfile, data_hr, label_label)
                iter=iter+1
        print(iter)

if __name__=="__main__":
    tran1=ToTensor()
    filename = "test_set.h5" #保存的文件名
    HR_dir = "../../B100"     #数据来源
    pfile = h5py.File(filename)
    main()
    pfile.close()

