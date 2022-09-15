import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_x_idx_val = x[idx]

        x[idx] = float(tmp_x_idx_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] =  float(tmp_x_idx_val) - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_x_idx_val # 还原值
        
    return grad
    
def function_2(x):
    #return x[0]**2+x[1]**2
    return np.sum(x**2)

def gradient_descent(f,init_x , lr=0.01,step_num=100):
    x=init_x
    for i in range(step_num):
        grad=numerical_gradient(f,x)
        x-=lr*grad

    return x

if __name__ == '__main__':
    grad1 = numerical_gradient(function_2, np.array([3.0, 4.0]) )
    print(grad1)
    #grad2 = numerical_gradient(function_2, np.array([0.0, 2.0]) )
    #print(grad2)
    #grad3 = numerical_gradient(function_2, np.array([3.0, 0.0]) )
    #print(grad3)
    #init_x=np.array([3.0,4.0])
    #min=gradient_descent(function_2,init_x,lr=0.1)
    #print(min)