import numpy as np
import matplotlib
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
#实时画图程序

myfont=fm.FontProperties(fname="/home/laowang/fonts/simsun.ttc",size=14)

def draw(list_x,x_length,list_y,y_length,data={}):
    #x轴为list_x,y轴为list_y,x_length和y_length分别为x轴和y轴的刻度个数,data为图例中的数据

    # 打开交互模式
    plt.ion()

    # 清除原有图像
    plt.cla()

    # 设定标题等
    plt.title("LOSS动态曲线图",fontproperties=myfont)
    plt.grid(True)

    # 生成测试数据
    x_tick = [i for i in list_x if list_x.index(i)%(len(list_x)//x_length+1)==0 ]
    y_tick = list_y[len(list_y)-10:len(list_y)]

    # 设置X轴
    plt.xlabel("X轴", fontproperties=myfont)
    plt.xlim(np.min(x_tick)-1, np.max(x_tick)+1)
    plt.xticks(x_tick)

    # 设置Y轴
    plt.ylabel("Y轴", fontproperties=myfont)
    plt.ylim(np.min(y_tick)-10, np.max(y_tick)+10)
    plt.yticks(y_tick)

    # 画曲线
    plt.plot(list_x, list_y, "r-", linewidth=2.0, label="MSE："+str(data))

    # 设置图例位置,loc可以为[upper, lower, left, right, center]
    plt.legend(loc="upper left", shadow=True)

    # 暂停
    plt.pause(1)

    # 关闭交互模式
    plt.ioff()

    return
if __name__=="__main__":
    list_y=[[1,2,4,5],[1,2,6,9,7],[5,5,6,4,7]]
    plt.figure(figsize=(8, 6), dpi=80)
    for list in list_y:
        draw(range(len(list)),10,list,10,{"laowang":123})
    plt.show()