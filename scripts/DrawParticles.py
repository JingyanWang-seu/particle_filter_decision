import numpy as np
import matplotlib.pyplot as plt



def draw_pars(x, y, scatter):

    plt.ion()

    # 更新散点图
    scatter.set_offsets(np.c_[x, y])  # 更新散点位置
    
    # 重绘图像
    plt.draw()
    plt.pause(0.1)  # 暂停 0.1 秒

# 保持图像显示
    plt.ioff()


