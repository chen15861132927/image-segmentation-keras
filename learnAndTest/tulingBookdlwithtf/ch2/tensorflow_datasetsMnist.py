import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_datasets as tfds

num_epoch = 5
batch_size = 50
learning_rate = 0.001

class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=200, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):         # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)    # [batch_size, 784]
        x = self.dense1(x)          # [batch_size, 200]
        x = self.dense2(x)          # [batch_size, 100]
        x = self.dense3(x)          # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output

IMAGE_WIDTH= IMAGE_HEIGHT=224

mnist_train = tfds.load(name="mnist", split="train", as_supervised=True)

model = MLP()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

num_batches = 1000#int(data_loader.num_train_data // batch_size * num_epochs)

sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
tileC=tf.constant([1,1,3])
mnist_train = mnist_train.map(lambda img, label: (tf.image.resize(tf.tile(img,tileC), (224, 224)) / 255.0, label)).shuffle(1024).batch(batch_size)
batch_index=0
#for batch_index in range(num_batches):
for images, labels in mnist_train:
    imagesnumpy=images.numpy()
    labelsnumpy=labels.numpy()

    # img3=tf.tile(images,tileC)
    # imagess=tf.image.resize(img3, (224, 224))
    # for xtindex in range(0,imagesnumpy.shape[0]):
        # imagess = imagesnumpy[xtindex] *255 # 把图像的形状变为原来的尺寸
        # pil_img = Image.fromarray(np.uint8(imagess),mode='RGB')
        # plt.imshow(pil_img) # 显示图片
        # plt.axis('off') # 不显示坐标轴
        # plt.show()
    
    with tf.GradientTape() as tape:
        y_pred = model(X)
        ynum=y_pred.numpy()
        #loss = tf.keras.losses.categorical_crossentropy(\
        #            y_true=tf.one_hot(y, depth=tf.shape(y_pred)[-1]),\
        #            y_pred=y_pred\
        #        )
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        lossnum=loss.numpy()
        loss = tf.reduce_logsumexp(loss)
        if(batch_index%batch_size==0):
            print("batch %d: loss %f" % (batch_index, loss.numpy()),end=' ')
            sparse_categorical_accuracy.update_state(y_true=y, y_pred=y_pred)
            print("test accuracy: %f" % sparse_categorical_accuracy.result())
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

# num_batches = int(data_loader.num_test_data // batch_size)
# for batch_index in range(num_batches):
#     start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
#     y_pred = model.predict(data_loader.test_data[start_index: end_index])
#     sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
# print("test accuracy: %f" % sparse_categorical_accuracy.result())