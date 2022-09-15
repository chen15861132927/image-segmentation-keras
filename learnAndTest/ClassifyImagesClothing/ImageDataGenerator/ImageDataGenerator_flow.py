#ImageDataGenerator.py

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
        
l=[]
img = load_img(r"img1.jpg")  # this is a PIL image
img1 = load_img(r"IMG_1626.jpg")  # this is a PIL image
y=img_to_array(img1)
x = img_to_array(img) # this is a Numpy array with shape (3, 150, 150)
l.append(x)
l.append(y)
l=np.array(l)
#l = l.reshape((1,) + l.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
 
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(l, batch_size=2,
                          save_to_dir=r'pic', save_format='jpg'):
    i += 1
    if i > 2:
        break  # otherwise the generator would loop indefinitely
