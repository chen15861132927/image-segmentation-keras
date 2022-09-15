# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np


X=np.random.rand(2)
W=np.random.rand(2,3)
B=np.random.rand(3)



XW=np.dot(X,W)

print(X)
print(W)
print(B)
print(XW)

print(X.shape)
print(W.shape)
print(B.shape)
print(XW.shape)

print(X.shape[0])
print(W.shape[0])
print(W.shape[1])
print(B.shape[0])
print(XW.shape[0])