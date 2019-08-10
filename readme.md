本项目为基于Pytorch的VDSR的重建


训练集为H5文件，根据原论文，应制作大约9000*64张尺寸41的单通道数据，，本网络使用的是3通道，故数量上再除以3，数据集大小为7GB

./data 中有数据集制作脚本，基于H5py，可在python环境中制作

./data/HR_2_LR.py    将高分辨率图像经过下采样和上采样后转换为同尺寸的第分辨率图像，输出与输入同目录

./data/set_made.py   制作H5文件的数据集和测试集，输出文件的标签为 /train/data  和/train/label 分别为数据和标签

./look_set.py     数据集制完成后使用该工具抽样检查数据集是否符合预期


draw.py  动态显示训练过程中的损失曲线， 目前显示并不是很完美，求改善

jilu.txt    训练过程是先用100张图片将模型预训练一遍，再用大数据集训练，以达到模型参数完美初始化的目的 该文件中记录了训练过程

main_train.py  训练可使用    python main_train.py --cuda

vdsr的代码参考了 @https://github.com/twtygqyy/pytorch-vdsr的代码， 将其单通道改为了3通道

PSNR_evalu.py  评估训测试的PSNR 测试集需要自己制作成H5文件   （刚开始用单文件直接输入，但后来发现，改文件名好麻烦）

PSNR.py   PSNR的计算代码，输入为两个图像

Input2Output.py   将数据低分辨率图片转换为高分辨率图片，尺寸不会变，所以需要自行使用./data/upscale.py工具放大尺寸

Set5文件夹内为测试使用的原图

set5input文件夹内为将Set5的原图下采样再上采样得到的模糊图

set5outout文件夹内为   以setinout为输入  使用Input2Output.py工具转换得到


欢迎指出问题

查阅资料一直用英文，，，，，，这次写中文


