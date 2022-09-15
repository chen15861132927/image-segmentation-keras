# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# img = x_train[0]
# label = t_train[0]
# print(label)  # 5

# print(img.shape)  # (784,)
# img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
# print(img.shape)  # (28, 28)

IMAGE_ROW = 20  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 20  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_WIDTH= IMAGE_HEIGHT=28
# pil_img = Image.fromarray(np.uint8(x_train[0].reshape(IMAGE_WIDTH, IMAGE_HEIGHT)))

# pastedImage = Image.new(pil_img.mode, (IMAGE_ROW*IMAGE_HEIGHT, IMAGE_COLUMN*IMAGE_WIDTH))
# labResult=[]

# for y in range(0, IMAGE_ROW):
#     for x in range(0, IMAGE_COLUMN):
#         index=y*IMAGE_COLUMN+x
#         img = x_train[index].reshape(IMAGE_WIDTH, IMAGE_HEIGHT)  # 把图像的形状变为原来的尺寸
#         pil_img = Image.fromarray(np.uint8(img))
#         #pil_img.show()
#         pastedImage.paste(pil_img, box=(x*IMAGE_WIDTH, y * IMAGE_HEIGHT))
#         labResult.append(t_train[index])


# img_show(pastedImage)
# print(labResult)

pil_img = Image.fromarray(np.uint8(x_test[0].reshape(IMAGE_WIDTH, IMAGE_HEIGHT)))

pastedImage = Image.new(pil_img.mode, (IMAGE_ROW*IMAGE_HEIGHT, IMAGE_COLUMN*IMAGE_WIDTH))
labResult=np.empty([IMAGE_ROW,IMAGE_COLUMN], dtype = int)

for y in range(0, IMAGE_ROW):
    for x in range(0, IMAGE_COLUMN):
        index=y*IMAGE_COLUMN+x
        img = x_test[index].reshape(IMAGE_WIDTH, IMAGE_HEIGHT)  # 把图像的形状变为原来的尺寸
        pil_img = Image.fromarray(np.uint8(img))
        #pil_img.show()
        pastedImage.paste(pil_img, box=(x*IMAGE_WIDTH, y * IMAGE_HEIGHT))
        labResult[y][x]=t_test[index]

print(labResult)
img_show(pastedImage)

