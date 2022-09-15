#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

# 模型参数
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
class LetNet:
    # 模型版本
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    subtract_pixel_mean=False
    model=None
    save_dir=None
    Root_Path=None
    x_train_mean=None
    def __init__(self,subtract_pixel_mean=False):
        self.subtract_pixel_mean=subtract_pixel_mean
        
    def compile(self,input_shape):
        self.model=self.letNetModel(input_shape=input_shape)
        #self.model.compile(loss='categorical_crossentropy',
        #            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr_schedule(0)),
        #            metrics=['acc'])
        # self.model.compile(optimizer='adam',
        #         loss=tf.keras.losses.CategoricalCrossentropy(),
        #         metrics=['acc'])
        self.model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['acc'])
        self.model.summary()



    def train(self,x_train, y_train,x_test, y_test,epochs=200,batch_size=32,Nodata_augmentation=True,modelFile=""):
        # 准备模型保存路径。
        self.save_dir = os.path.join(self.Root_Path, 'LearnLetNetMnistSaved_models')
        # 如果使用减去像素均值
        if self.subtract_pixel_mean:
            self.x_train_mean = np.mean(x_train, axis=0)
            x_train -= self.x_train_mean
            x_test -= self.x_train_mean
        model_name = 'LetNetMnist_model-epoch{epoch:04d}-acc{acc:.03f}-val_acc{val_acc:.04f}.h5' 
        if not os.path.isdir(self.save_dir ):
            os.makedirs(self.save_dir )
        filepath = os.path.join(self.save_dir , model_name)
        #print(filepath)

        loadfile = self.save_dir +modelFile
        if os.path.isfile(loadfile):
            self.model = tf.keras.models.load_model(loadfile)

        # 准备保存模型和学习速率调整的回调。
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                        monitor='val_acc',
                                                        verbose=1,
                                                        save_best_only=True)
        callbacks = [checkpoint]

        # 运行训练，是否数据增强可选。
        if Nodata_augmentation:
            print('Not using data augmentation.')
            self.model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=callbacks)
        else:
            print('Using real-time data augmentation.')
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
                rotation_range=0,
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
                horizontal_flip=True,
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
            datagen.fit(x_train)

            # 在由 datagen.flow() 生成的批次上拟合模型。
            self.model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_test, y_test),
                    epochs=epochs, verbose=1, workers=4,
                    callbacks=callbacks)

    def trainWithOutTest(self,x_train, y_train,epochs=200,batch_size=32,Nodata_augmentation=True,modelFile=""):
        
        if(isinstance(y_train, tf.Tensor)):
            y_train=y_train.numpy()
        # 准备模型保存路径。
        self.save_dir = os.path.join(self.Root_Path, 'LearnLetNetMnistSaved_models')
        # 如果使用减去像素均值
        if self.subtract_pixel_mean:
            self.x_train_mean = np.mean(x_train, axis=0)
            x_train -= self.x_train_mean
           
        model_name = 'LetNetMnist%s_modelcp-{epoch:04d}-{val_acc:.04f}.h5' 
        if not os.path.isdir(self.save_dir ):
            os.makedirs(self.save_dir )
        filepath = os.path.join(self.save_dir , model_name)
        print(filepath)

        loadfile = self.save_dir +modelFile
        if os.path.isfile(loadfile):
            self.model = tf.keras.models.load_model(loadfile)

        # 准备保存模型和学习速率调整的回调。
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                        monitor='val_acc',
                                                        verbose=1,
                                                        save_best_only=True)
        callbacks = [checkpoint]

        # 运行训练，是否数据增强可选。
        if Nodata_augmentation:
            print('Not using data augmentation.')
            self.model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True)
        else:
            print('Using real-time data augmentation.')
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
                rotation_range=0,
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
                horizontal_flip=True,
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
            datagen.fit(x_train)

            # 在由 datagen.flow() 生成的批次上拟合模型。
            self.model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs, verbose=1, workers=4,
                    callbacks=callbacks)

    def letNetModel(self,input_shape,  num_classes=10):
        # model = tf.keras.Sequential([
        #     # 卷积层1
        #     tf.keras.layers.Conv2D(6, (5,5), input_shape=input_shape),  # 使用6个5*5的卷积核对单通道32*32的图片进行卷积，结果得到6个28*28的特征图
        #     tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),  # 对28*28的特征图进行2*2最大池化，得到14*14的特征图
        #     tf.keras.layers.ReLU(),  # ReLU激活函数
        #     # 卷积层2
        #     tf.keras.layers.Conv2D(16, (5,5), input_shape=input_shape),  # 使用16个5*5的卷积核对6通道14*14的图片进行卷积，结果得到16个10*10的特征图
        #     tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),  # 对10*10的特征图进行2*2最大池化，得到5*5的特征图
        #     tf.keras.layers.ReLU(),  # ReLU激活函数
        #     # 卷积层3
        #     tf.keras.layers.Conv2D(120, (5,5), input_shape=input_shape),  # 使用120个5*5的卷积核对16通道5*5的图片进行卷积，结果得到120个1*1的特征图
        #     tf.keras.layers.ReLU(),  # ReLU激活函数
        #     # 将 (None, 1, 1, 120) 的下采样图片拉伸成 (None, 120) 的形状
        #     tf.keras.layers.Flatten(input_shape=input_shape),
        #     # 全连接层1
        #     tf.keras.layers.Dense(84, activation='relu'),  # 120*84
        #     # 全连接层2
        #     tf.keras.layers.Dense(num_classes, activation='softmax')  # 84*10
        # ])
        model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(6, (5,5), padding="same",activation='relu', input_shape=input_shape),
                    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                    tf.keras.layers.ReLU(),  # ReLU激活函数
                    tf.keras.layers.Conv2D(16, (5,5), padding="same",activation='relu', input_shape=input_shape),
                    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                    tf.keras.layers.ReLU(),  # ReLU激活函数
                    tf.keras.layers.Conv2D(120, (5,5), padding="same",activation='relu', input_shape=input_shape),
                    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                    tf.keras.layers.ReLU(),  # ReLU激活函数
                    tf.keras.layers.Flatten(input_shape=input_shape),
                    tf.keras.layers.Dense(84, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])
        model.build(input_shape)
        #model.summary()
        
        return model

    def evaluateAndSaveFail(self,x,y,batch_size=32,evaluateName="test"):
        # testscores = self.model.evaluate(x, y, verbose=1)
        # print(evaluateName+' loss:', testscores[0])
        # print(evaluateName+' accuracy:', testscores[1])
        x_Origintest=np.copy(x)*255
        # 如果使用减去像素均值
        if self.subtract_pixel_mean:
            x -= self.x_train_mean

        probability_model_test = tf.keras.Sequential(
            [self.model, tf.keras.layers.Softmax()])
        predictions_test = probability_model_test.predict(x)

        print(evaluateName+" calculating accuracy ... ")

        IMAGE_WIDTH = IMAGE_HEIGHT = 28
        test_failcount = 0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            origintx = x_Origintest[i*batch_size:(i+1)*batch_size]
            originy = y[i*batch_size:(i+1)*batch_size] #np.argmax(y[i*batch_size:(i+1)*batch_size], axis=1)
            predictionsy = np.argmax(predictions_test[i*batch_size:(i+1)*batch_size], axis=1)

            for inotsame, val in enumerate(predictionsy == originy):
                if not val:
                    test_failcount = test_failcount+1
                    path = self.save_dir +"\\"+evaluateName+"\\"
                    if(os.path.exists(path) == False):
                        os.makedirs(path)
                    imgarray = tx[inotsame].reshape(IMAGE_WIDTH, IMAGE_HEIGHT) 
                    imgarray = imgarray*255
                    imgarray= np.uint8(imgarray)
                    originimgarray = origintx[inotsame].reshape(IMAGE_WIDTH, IMAGE_HEIGHT) 
                    pil_img = Image.fromarray(np.uint8(originimgarray))
                    filename = str(i)+"_"+str(inotsame)+"Ori" + \
                        str(originy[inotsame])+"_Pre"+str(predictionsy[inotsame])
                    print(str(test_failcount)+"  "+filename)
                    pil_img.save(path+filename+".jpg", "JPEG")

        accuracy=1.0-1.0*test_failcount/x.shape[0]
        print(evaluateName+' accuracy:', accuracy)

