import tensorflow as tf
 
version = tf.__version__
gpu_ok = tf.test.is_gpu_available()
print("tf version:", version, "\nuse GPU", gpu_ok)

import tensorflow as tf
import timeit

with tf.device('/cpu:1'):
    cpu_a = tf.random.normal([10000, 1000])
    cpu_b = tf.random.normal([1000, 2000])
    print(cpu_a.device, cpu_b.device)

with tf.device('/cpu:0'):
    gpu_a = tf.random.normal([10000, 1000])
    gpu_b = tf.random.normal([1000, 2000])
    print(gpu_a.device, gpu_b.device)

def cpu_run():
    with tf.device('/cpu:1'):
        c = tf.matmul(cpu_a, cpu_b)
    return c

def gpu_run():
    with tf.device('/cpu:0'):
        c = tf.matmul(gpu_a, gpu_b)
    return c

# warm up	这里就当是先给gpu热热身了
cpu_time = timeit.timeit(cpu_run, number=100)
gpu_time = timeit.timeit(gpu_run, number=100)
print('warmup:', cpu_time, gpu_time)
