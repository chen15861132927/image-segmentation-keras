import os
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

cwd = os.getcwd()
print(cwd)
Root_Path=sys.path[0]
googlePath="/content/drive/MyDrive/"
print(os.path.isdir(googlePath))
if(cwd.startswith("/content")):
  if(os.path.isdir(googlePath) is False):
    from google.colab import drive
    drive.mount('/content/drive')
  Root_Path = os.path.join(cwd,googlePath)
print(Root_Path) 


# 训练参数
batch_size = 2  # 原论文按照 batch_size=128 训练所有的网络
epochs = 2
num_classes = 10

# 载入 mnist 数据。
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 输入图像维度。
x_train = np.expand_dims(x_train, axis = 3)
x_test = np.expand_dims(x_test, axis = 3)

# 数据标准化。
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 

print('x_train shape:', x_train.shape,x_train.shape[0], 'train samples')
print('x_test shape:', x_test.shape,x_test.shape[0], 'test samples')

# 将类向量转换为二进制类矩阵。
# y_train = tf.keras.utils.to_categorical(y_train, num_classes)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# 这将做预处理和实时数据增强。
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    # 在整个数据集上将输入均值置为 0
    featurewise_center=False,
    # 将每个样本均值置为 0
    samplewise_center=False,
    # 将输入除以整个数据集的 std
    featurewise_std_normalization=False,
    # 将每个输入除以其自身 std
    samplewise_std_normalization=False,
    # 应用 ZCA 白化
    zca_whitening=False,
    # ZCA 白化的 epsilon 值
    zca_epsilon=1e-06,
    # 随机图像旋转角度范围 (deg 0 to 180)
    rotation_range=30,
    # 随机水平平移图像
    width_shift_range=0.1,
    # 随机垂直平移图像
    height_shift_range=0.1,
    # 设置随机裁剪范围
    shear_range=0.,
    # 设置随机缩放范围
    zoom_range=0.,
    # 设置随机通道切换范围
    channel_shift_range=0.,
    # 设置输入边界之外的点的数据填充模式
    fill_mode='nearest',
    # 在 fill_mode = "constant" 时使用的值
    cval=0.,
    # 随机翻转图像
    horizontal_flip=False,
    # 随机翻转图像
    vertical_flip=False,
    # 设置重缩放因子 (应用在其他任何变换之前)
    rescale=None,
    # 设置应用在每一个输入的预处理函数
    preprocessing_function=None,
    # 图像数据格式 "channels_first" 或 "channels_last" 之一
    data_format=None,
    # 保留用于验证的图像的比例 (严格控制在 0 和 1 之间)
    validation_split=0.0)
# 计算大量的特征标准化操作
# (如果应用 ZCA 白化，则计算 std, mean, 和 principal components)。
#第三步：对需要处理的数据进行fit
datagenx_train=x_train
datagen.fit(datagenx_train)


#第四步：使用.flow方法构造Iterator， 
data_iter = datagen.flow(datagenx_train, y_train, batch_size=batch_size)  #返回的是一个“生成器对象”
print(type(data_iter))    #返回的是一个NumpyArrayIterator对象
 

epochsIndex=0
totalcount=0
path=os.path.join(Root_Path,"FlowPath")
if(os.path.exists(path) == False):
  os.makedirs(path)
IMAGE_WIDTH = IMAGE_HEIGHT = 28
# 通过循环迭代每一次的数据，并进行查看
for x_batch,y_batch in data_iter:
  #print(epochsIndex)
  epochsIndex=epochsIndex+1

    # for i in range(8):
    #     plt.subplot(2,4,i+1)
    #     plt.imshow(x_batch[i].reshape(28,28), cmap='gray')
    # plt.show()  
  print("totalcount:",totalcount," epochsIndex:",epochsIndex)
  totalcount=totalcount+x_batch.shape[0]
  tx = x_batch
  tt = y_batch
  for batchindex in range(int(x_batch.shape[0])):
    print("totalcount:",totalcount," epochsIndex:",epochsIndex," batchindex:",batchindex)
    imgarray = x_batch[batchindex]
    imgarray=imgarray.reshape(IMAGE_WIDTH, IMAGE_HEIGHT) 
    pil_img = Image.fromarray(np.uint8(imgarray))
    tempPath=os.path.join(path,str(y_batch[batchindex]))
    if(os.path.exists(tempPath) == False):
      os.makedirs(tempPath)
    filename = str(y_batch[batchindex])+"_"+str(epochsIndex)+"_"+str(batchindex)
    fileSavePath=os.path.join(tempPath,filename+".jpg")
    pil_img.save(fileSavePath, "JPEG")
    totalcount=totalcount+1

  break
