import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
x = tf.Variable(initial_value=3.)
m=x.numpy()
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.pow(x,5)
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print(y, y_grad)

print(m, y_grad.numpy())