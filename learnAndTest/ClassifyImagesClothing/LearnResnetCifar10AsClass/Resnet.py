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
class Resnet:
    # 模型版本
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = 2
    n = 3
    depth=1
    subtract_pixel_mean=False
    model=None
    model_type=None
    save_dir=None
    Root_Path=None
    x_train_mean=None
    def __init__(self,version=2,n=3,subtract_pixel_mean=False):

        self.version=version
        self.n=n
        # 从提供的模型参数 n 计算的深度
        if self.version == 1:
            self.depth = self.n * 6 + 2
        elif self.version == 2:
            self.depth = self.n * 9 + 2

        self.subtract_pixel_mean=subtract_pixel_mean
    def compile(self,input_shape):
        self.model=self.resnet_v2(input_shape=input_shape, depth=self.depth)
        if self.version == 1:
            self.model=self.resnet_v1(input_shape=input_shape, depth=self.depth)

        self.model.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr_schedule(0)),
                    metrics=['acc'])
        self.model.summary()

    def train(self,x_train, y_train,x_test, y_test,epochs=200,batch_size=32,Nodata_augmentation=True,datasetsName='Cifar10',modelFile=""):
        # 模型名称、深度和版本
        self.model_type = 'ResNet%dv%d' % (self.depth,  self.version)
        # 准备模型保存路径。
        self.save_dir = os.path.join(self.Root_Path, 'LearnResNet'+datasetsName+'Saved_models'+ self.model_type)
        # 如果使用减去像素均值
        if self.subtract_pixel_mean:
            self.x_train_mean = np.mean(x_train, axis=0)
            x_train -= self.x_train_mean
            x_test -= self.x_train_mean
        model_name = 'ResNet'+datasetsName+'%s_model-{epoch:04d}-{val_acc:.04f}.h5' % self.model_type
        if not os.path.isdir(self.save_dir ):
            os.makedirs(self.save_dir )
        filepath = os.path.join(self.save_dir , model_name)
        print(filepath)

        loadfile = self.save_dir +'//'+modelFile
        if os.path.isfile(loadfile):
            self.model = tf.keras.models.load_model(loadfile)

        # 准备保存模型和学习速率调整的回调。
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                        monitor='val_acc',
                                                        verbose=1,
                                                        save_best_only=True)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(self.lr_schedule)

        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                        cooldown=0,
                                                        patience=5,
                                                        min_lr=0.5e-6)

        callbacks = [checkpoint, lr_reducer, lr_scheduler]

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



    def lr_schedule(self,epoch):
        """学习率调度

        学习率将在 80, 120, 160, 180 轮后依次下降。
        他作为训练期间回调的一部分，在每个时期自动调用。

        # 参数
            epoch (int): 轮次

        # 返回
            lr (float32): 学习率
        """
        lr = 1e-3
        if epoch > 200:
            lr *= 1e-4
        elif epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    def resnet_layer(self,inputs,
                    num_filters=16,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    batch_normalization=True,
                    conv_first=True):
        """2D 卷积批量标准化 - 激活栈构建器

        # 参数
            inputs (tensor): 从输入图像或前一层来的输入张量
            num_filters (int): Conv2D 过滤器数量
            kernel_size (int): Conv2D 方形核维度
            strides (int): Conv2D 方形步幅维度
            activation (string): 激活函数名
            batch_normalization (bool): 是否包含批标准化
            conv_first (bool): conv-bn-activation (True) 或
                bn-activation-conv (False)

        # 返回
            x (tensor): 作为下一层输入的张量
        """
        conv = tf.keras.layers.Conv2D(num_filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding='same',
                                    kernel_initializer='he_normal',
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            if activation is not None:
                x = tf.keras.layers.Activation(activation)(x)
        else:
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            if activation is not None:
                x = tf.keras.layers.Activation(activation)(x)
            x = conv(x)
        return x

    def resnet_v1(self,input_shape, depth, num_classes=10):
        """ResNet 版本 1 模型构建器 [a]

        2 x (3 x 3) Conv2D-BN-ReLU 的堆栈
        最后一个 ReLU 在快捷连接之后。
        在每个阶段的开始，特征图大小由具有 strides=2 的卷积层减半（下采样），
        而滤波器的数量加倍。在每个阶段中，这些层具有相同数量的过滤器和相同的特征图尺寸。
        特征图尺寸:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        参数数量与 [a] 中表 6 接近:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M

        # 参数
            input_shape (tensor): 输入图像张量的尺寸
            depth (int): 核心卷积层的数量
            num_classes (int): 类别数 (CIFAR10 为 10)

        # 返回
            model (Model): Keras 模型实例
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # 开始模型定义
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = tf.keras.layers.Input(shape=input_shape)
        x = self.resnet_layer(inputs=inputs)
        # 实例化残差单元的堆栈
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # 第一层但不是第一个栈
                    strides = 2  # downsample
                y = self.resnet_layer(inputs=x,
                                num_filters=num_filters,
                                strides=strides)
                y = self.resnet_layer(inputs=y,
                                num_filters=num_filters,
                                activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # 线性投影残差快捷键连接，以匹配更改的 dims
                    x = self.resnet_layer(inputs=x,
                                    num_filters=num_filters,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=None,
                                    batch_normalization=False)
                x = tf.keras.layers.add([x, y])
                x = tf.keras.layers.Activation('relu')(x)
            num_filters *= 2

        # 在顶层加分类器。
        # v1 不在最后一个快捷连接 ReLU 后使用 BN
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        y = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(num_classes,
                                        activation='softmax',
                                        kernel_initializer='he_normal')(y)

        # 实例化模型。
        model=tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return model

    def resnet_v2(self,input_shape, depth, num_classes=10):
        """ResNet 版本 2 模型构建器 [b]

        (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D 的堆栈，也被称为瓶颈层。
        每一层的第一个快捷连接是一个 1 x 1 Conv2D。
        第二个及以后的快捷连接是 identity。
        在每个阶段的开始，特征图大小由具有 strides=2 的卷积层减半（下采样），
        而滤波器的数量加倍。在每个阶段中，这些层具有相同数量的过滤器和相同的特征图尺寸。
        特征图尺寸:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256

        # 参数
            input_shape (tensor): 输入图像张量的尺寸
            depth (int): 核心卷积层的数量
            num_classes (int): 类别数 (CIFAR10 为 10)

        # 返回
            model (Model): Keras 模型实例
        """
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # 开始模型定义。
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)

        inputs = tf.keras.layers.Input(shape=input_shape)
        # v2 在将输入分离为两个路径前执行带 BN-ReLU 的 Conv2D 操作。
        x = self.resnet_layer(inputs=inputs,
                        num_filters=num_filters_in,
                        conv_first=True)

        # 实例化残差单元的栈
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2    # downsample

                # 瓶颈残差单元
                y = self.resnet_layer(inputs=x,
                                num_filters=num_filters_in,
                                kernel_size=1,
                                strides=strides,
                                activation=activation,
                                batch_normalization=batch_normalization,
                                conv_first=False)
                y = self.resnet_layer(inputs=y,
                                num_filters=num_filters_in,
                                conv_first=False)
                y = self.resnet_layer(inputs=y,
                                num_filters=num_filters_out,
                                kernel_size=1,
                                conv_first=False)
                if res_block == 0:
                    # 线性投影残差快捷键连接，以匹配更改的 dims
                    x = self.resnet_layer(inputs=x,
                                    num_filters=num_filters_out,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=None,
                                    batch_normalization=False)
                x = tf.keras.layers.add([x, y])

            num_filters_in = num_filters_out

        # 在顶层添加分类器
        # v2 has BN-ReLU before Pooling
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        y = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(num_classes,
                                        activation='softmax',
                                        kernel_initializer='he_normal')(y)

        # 实例化模型。
        model=tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
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
            tt = np.argmax(y[i*batch_size:(i+1)*batch_size], axis=1)
            predictionsy = np.argmax(predictions_test[i*batch_size:(i+1)*batch_size], axis=1)

            for inotsame, val in enumerate(predictionsy == tt):
                if not val:
                    test_failcount = test_failcount+1
                    path = self.save_dir +"//"+evaluateName+"//"
                    if(os.path.exists(path) == False):
                        os.makedirs(path)
                    imgarray = tx[inotsame].reshape(IMAGE_WIDTH, IMAGE_HEIGHT) 
                    imgarray = imgarray*255
                    imgarray= np.uint8(imgarray)
                    originimgarray = origintx[inotsame].reshape(IMAGE_WIDTH, IMAGE_HEIGHT) 
                    pil_img = Image.fromarray(np.uint8(originimgarray))
                    filename = str(i)+"_"+str(inotsame)+"R" + \
                        str(tt[inotsame])+"_W"+str(predictionsy[inotsame])
                    print(str(test_failcount)+"  "+filename)
                    pil_img.save(path+filename+".jpg", "JPEG")

        accuracy=1.0-1.0*test_failcount/x.shape[0]
        print(evaluateName+' accuracy:', accuracy)

