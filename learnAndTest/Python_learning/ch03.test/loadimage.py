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

s = b'hello world'

res=np.frombuffer(s, dtype='S1', count=5, offset=6) 
print(res)
np.array([b'w', b'o', b'r', b'l', b'd'], dtype='|S1')

res2=np.frombuffer(b'\x01\x02', dtype=np.uint8)
print(res2)
np.array([1, 2], dtype=np.uint8) 

res3=np.frombuffer(b'\x01\x02\x03\x04\x05', dtype=np.uint8, count=3)
print(res3)
np.array([1, 2, 3], dtype=np.uint8)

_load_img("C:\\Python_learning\\ch03\\train-images-idx3-ubyte.gz")