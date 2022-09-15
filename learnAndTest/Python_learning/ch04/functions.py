# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def origincross_entropy_error(y, t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))



def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)



# y =np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])
# t =np.array([0  ,0   ,1  ,  0,   0,  0,  0,  0,  0,  0])


# y2=np.array([0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0])


y =np.array([0  ,0   ,0.01,  0,   0,  0,  0,  0,  0,  0])
t =np.array([0  ,0   ,0   ,  0,   0,  0,  0,  1,  0,  0])


y2=np.array([0.9,0.9,0.9,0.9,0.9,0.9,0.9,1,0.9,0.9])



differ1=mean_squared_error(y,t)
# print((y-t))
# print((y-t)**2)
# print(np.sum((y-t)**2))
# print(0.5*np.sum((y-t)**2))
print(differ1)
differ2=mean_squared_error(y2,t)
# print((y2-t))
# print((y2-t)**2)
# print(np.sum((y2-t)**2))
# print(0.5*np.sum((y2-t)**2))
print(differ2)


delta=1e-7
crossdiffer1=cross_entropy_error(y,t)

print(np.log(y+delta))

print(t*np.log(y+delta))

print(crossdiffer1)

crossdiffer2=cross_entropy_error(y2,t)

print(np.log(y2+delta))

print(t*np.log(y2+delta))


print(crossdiffer2)