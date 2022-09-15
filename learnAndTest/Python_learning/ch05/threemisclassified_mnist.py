# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt # plt 用于显示图片
import datetime
import numpy as np
from mnist import load_mnist
from three_layer_net import ThreeLayerNet
from PIL import Image

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
hidden_size=200
network = ThreeLayerNet(input_size=784, hidden_size1=hidden_size, hidden_size2=int(hidden_size/2), output_size=10)

fileName="C:\Python_learning\ch05\params997_999.pkl"
network.load_params()

print("calculating test accuracy ... ")
#sampled = 1000
#x_test = x_test[:sampled]
#t_test = t_test[:sampled]

test_acc = 0.0
batch_size = 100
IMAGE_WIDTH= IMAGE_HEIGHT=28
test_failcount=0
for i in range(int(x_test.shape[0] / batch_size)):
    tx = x_test[i*batch_size:(i+1)*batch_size]
    tt = t_test[i*batch_size:(i+1)*batch_size]
    y = network.predict(tx)
    y = np.argmax(y, axis=1)
    tres = np.argmax(tt, axis=1)
    for inotsame, val in enumerate(y == tres):
        if not val:
            test_failcount=test_failcount+1
            path=sys.path[0]+"\\x_test\\"
            if(os.path.exists(path)==False):
                os.makedirs(path) 
            imgarray=tx[inotsame]*255
            img = imgarray.reshape(IMAGE_WIDTH, IMAGE_HEIGHT)  # 把图像的形状变为原来的尺寸
            pil_img = Image.fromarray(np.uint8(img))
            filename=str(i)+"_"+str(inotsame)+"R"+str(tres[inotsame])+"_L"+str(y[inotsame])
            print(str(test_failcount)+"  "+filename)
            pil_img.save(path+filename+".jpg", "JPEG")

    test_acc += np.sum(y == tres)
    
test_acc = test_acc / x_test.shape[0]
print("test accuracy:" + str(test_acc))
print("calculating train accuracy ... ") 

train_acc= 0.0
batch_size = 100
train_failcount=0

for i in range(int(x_train.shape[0] / batch_size)):
    tx = x_train[i*batch_size:(i+1)*batch_size]
    tt = t_train[i*batch_size:(i+1)*batch_size]
    y = network.predict(tx)
    y = np.argmax(y, axis=1)
    tres = np.argmax(tt, axis=1)
    for inotsame, val in enumerate(y == tres):
        if not val:
            train_failcount=train_failcount+1
            path=sys.path[0]+"\\x_train\\"
            if(os.path.exists(path)==False):
                os.makedirs(path) 
            imgarray=tx[inotsame]*255        
            img = imgarray.reshape(IMAGE_WIDTH, IMAGE_HEIGHT)  # 把图像的形状变为原来的尺寸
            pil_img = Image.fromarray(np.uint8(img))
            filename=str(i)+"_"+str(inotsame)+"R"+str(tres[inotsame])+"_L"+str(y[inotsame])
            print(str(train_failcount)+"  "+filename)
            pil_img.save(path+filename+".jpg", "JPEG")

    train_acc += np.sum(y == tres)
    
train_acc = train_acc / x_train.shape[0]
print("train accuracy:" + str(train_acc))
