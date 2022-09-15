# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from mnist import load_mnist
from PIL import Image

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params("C:\Python_learning\ch08\deep_convnet_params.994.pkl")


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
    y = network.predict(tx, train_flg=False)
    y = np.argmax(y, axis=1)
    for inotsame, val in enumerate(y == tt):
        if not val:
            test_failcount=test_failcount+1
            path=sys.path[0]+"\\x_test\\"
            if(os.path.exists(path)==False):
                os.makedirs(path) 
            imgarray=tx[inotsame][0]*255
            pil_img = Image.fromarray(np.uint8(imgarray))
            filename=str(i)+"_"+str(inotsame)+"R"+str(tt[inotsame])+"_L"+str(y[inotsame])
            print(str(test_failcount)+"  "+filename)
            pil_img.save(path+filename+".jpg", "JPEG")

    test_acc += np.sum(y == tt)
    
test_acc = test_acc / x_test.shape[0]
print("test accuracy:" + str(test_acc))
print("calculating train accuracy ... ") 

train_acc= 0.0
batch_size = 100
train_failcount=0

for i in range(int(x_train.shape[0] / batch_size)):
    tx = x_train[i*batch_size:(i+1)*batch_size]
    tt = t_train[i*batch_size:(i+1)*batch_size]
    y = network.predict(tx, train_flg=False)
    y = np.argmax(y, axis=1)
    for inotsame, val in enumerate(y == tt):
        if not val:
            train_failcount=train_failcount+1
            path=sys.path[0]+"\\x_train\\"
            if(os.path.exists(path)==False):
                os.makedirs(path) 
            imgarray=tx[inotsame][0]*255
            pil_img = Image.fromarray(np.uint8(imgarray))
            filename=str(i)+"_"+str(inotsame)+"R"+str(tt[inotsame])+"_L"+str(y[inotsame])
            print(str(train_failcount)+"  "+filename)
            pil_img.save(path+filename+".jpg", "JPEG")

    train_acc += np.sum(y == tt)
    
train_acc = train_acc / x_train.shape[0]
print("train accuracy:" + str(train_acc))
