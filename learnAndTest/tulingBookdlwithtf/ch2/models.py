#Linear Regression
# 
# 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
import tensorflow as tf

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

xI = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
yO = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

X = tf.constant(xI)
Y = tf.constant(yO)
class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):     # 这里 input_shape 是第一次运行call()时参数inputs的形状
        self.w = self.add_weight(name='w',
            shape=[self.units], initializer=tf.zeros_initializer())
        self.b = self.add_weight(name='b',
            shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs):
        y_pred = inputs*self.w + self.b
        #print(inputs,self.w,y_pred,self.b)
        return y_pred

class LinearModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = LinearLayer(units=1)

    def call(self, inputs):
        output = self.layer(inputs)
        return output


# 以下代码结构与前节类似
model = LinearModel()
num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
for e in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = model(X)      # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
        loss = tf.square(y_pred - Y)
    grads = tape.gradient(loss, model.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

result=model(X) 
print(result.numpy())
print(model.variables)