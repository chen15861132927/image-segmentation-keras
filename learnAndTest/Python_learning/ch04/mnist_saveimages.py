# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from mnist import load_mnist
from PIL import Image
import matplotlib.pyplot as plt # plt 用于显示图片


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

IMAGE_WIDTH= IMAGE_HEIGHT=28

imgcount=x_test.shape[0]
for xtindex in range(0,imgcount ):
    img = x_test[xtindex].reshape(IMAGE_WIDTH, IMAGE_HEIGHT)  # 把图像的形状变为原来的尺寸
    pil_img = Image.fromarray(np.uint8(img))
    plt.imshow(pil_img) # 显示图片
    plt.axis('off') # 不显示坐标轴
    plt.show()
    print(xtindex)
    path=sys.path[0]+"\\x_test\\"+str(t_test[xtindex])
    if(os.path.exists(path)==False):
        os.makedirs(path) 
    pil_img.save(path+"\\"+str(xtindex)+"_"+str(t_test[xtindex])+".jpg", "JPEG")

