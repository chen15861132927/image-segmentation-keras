import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

num_epoch = 50
batch_size = 100
learning_rate = 0.01
IMAGE_WIDTH= IMAGE_HEIGHT=224
shuffleCount=2048
batch_index=0

dataset = tfds.load(name="mnist", split="train", as_supervised=True)

tileC=tf.constant([1,1,3])
dataset = dataset.map(lambda img, label: (tf.image.resize(tf.tile(img,tileC), (IMAGE_WIDTH, IMAGE_HEIGHT)) / 255.0, label))\
    .shuffle(shuffleCount).batch(batch_size)

model = tf.keras.applications.MobileNetV2(weights=None, classes=10)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

checkpoint = tf.train.Checkpoint(model=model)      
checkpoint.restore(tf.train.latest_checkpoint('./save'))
# checkpoint = tf.train.Checkpoint(model=model)      
manager = tf.train.CheckpointManager(checkpoint, directory='./save', max_to_keep=3)

sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

for e in range(num_epoch):
    for images, labels in dataset:
        with tf.GradientTape() as tape:
            labels_pred = model(images, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=labels_pred)
            loss = tf.reduce_mean(loss)
            if(batch_index%5==0):
                # 使用CheckpointManager保存模型参数到文件并自定义编号
                path = manager.save(checkpoint_number=batch_index)         
                print("batch %d: loss %f" % (batch_index, loss.numpy()),end=' ')
                sparse_categorical_accuracy.update_state(y_true=labels, y_pred=labels_pred)
                print("test accuracy: %f" % sparse_categorical_accuracy.result(),end=' ')
                print("model saved to %s" % path)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
        batch_index=batch_index+1
