# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt # plt 用于显示图片
import datetime
import numpy as np
from mnist import load_mnist
from two_layer_net import TwoLayerNet
from PIL import Image

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
hidden_size=300
network = TwoLayerNet(input_size=784, hidden_size=hidden_size, output_size=10)

iters_num = 20000
train_size = x_train.shape[0]
test_size = x_test.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
starttime = datetime.datetime.now()
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 梯度
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    #print("{},{}".format(i,loss))


    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

endtime = datetime.datetime.now()
RunTime=(endtime - starttime).seconds
print ("Run Time:{}".format(RunTime))

fileName="Train_R{0}_train_acc{1:3f}_test_acc{2:3f}_hidden_size{3}".format(RunTime,train_acc,test_acc,hidden_size)
np.save(fileName, network.params) 
eval_size=1
eval_mask = np.random.choice(train_size, eval_size)
x_eval = x_train[eval_mask]
t_eval = t_train[eval_mask]
print(np.argmax(t_eval))

y_eval=network.realpredict(x_eval)

IMAGE_WIDTH= IMAGE_HEIGHT=28

x_eval[0]= x_eval[0]* 255.0
img =x_eval[0].reshape(IMAGE_WIDTH, IMAGE_HEIGHT)  # 把图像的形状变为原来的尺寸
pil_img = Image.fromarray(np.uint8(img))

print(y_eval[0])
plt.imshow(pil_img) # 显示图片
plt.show()