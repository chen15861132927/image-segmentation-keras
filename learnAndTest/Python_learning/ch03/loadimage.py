import sys, os
import numpy as np
from PIL import Image
import gzip

def _load_img(file_name):
    img_size = 784

    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_name, 'rb') as f:
            data = np.frombuffer(f, np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data


_load_img("C:\\Python_learning\\ch03\\train-images-idx3-ubyte.gz")