# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
#from deep_convnet import DeepConvNet
from Ndeep_convnet import NDeepConvNet

from trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = NDeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='SGD', optimizer_param={'lr':0.01},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 保存参数
network.save_params("deep_nconvnet_params.pkl")
print("Saved Network Parameters!")
