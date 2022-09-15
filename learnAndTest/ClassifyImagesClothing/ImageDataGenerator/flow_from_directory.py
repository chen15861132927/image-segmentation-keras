from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
 
gener=datagen.flow_from_directory(r'C:\Dev\GithubProject\dlwithtf\ClassifyImagesClothing\ImageDataGenerator\train',#类别子文件夹的上一级文件夹
                                         batch_size=2,
                                         shuffle=False,
                                         target_size=(1024,1024),
                                         save_to_dir=r'train_result',
                                         save_prefix='trans_',
                                         save_format='jpg')

for i in range(10):
    gener.next()
