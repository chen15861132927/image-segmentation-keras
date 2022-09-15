import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

num_epoch = 5
batch_size = 50
learning_rate = 0.001

TRAIN = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)

TRAIN = TRAIN.map(lambda img, label: (tf.image.resize(img, (224, 224)) / 255.0, label)).shuffle(1024).batch(batch_size)

model = tf.keras.applications.MobileNetV2(weights=None, classes=5)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
batch_index=0

train_sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

for e in range(num_epoch):
    for images, labels in TRAIN:
        with tf.GradientTape() as tape:
            labels_pred = model(images, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=labels_pred)
            loss = tf.reduce_mean(loss)

            if(batch_index%5==0):
                print("batch %d: loss %f" % (batch_index, loss.numpy()),end=' ')
                train_sparse_categorical_accuracy.update_state(y_true=labels, y_pred=labels_pred)
                print("train accuracy: %f" % train_sparse_categorical_accuracy.result())
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
        batch_index=batch_index+1

