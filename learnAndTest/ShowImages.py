#!pip install git+https://github.com/divamgupta/image-segmentation-keras
#!pip install git+https://github.com/divamgupta/image-segmentation-keras
#!pip install git+https://github.com/chenweiSeagate/image-segmentation-keras
#! wget -P /content/drive/MyDrive/keras_segmentatio/dataset https://github.com/divamgupta/datasets/releases/download/seg/dataset1.zip && !unzip /content/drive/MyDrive/keras_segmentatio/dataset1.zip -d  /content/drive/MyDrive/keras_segmentatio/dataset 
#@title head { run: "auto", form-width: "150px" }
#region import
import os
import sys
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# plt.ion()
import tensorflow_datasets as tfds

#endregion
#region pwd Path and print current path
print(sys.version)
print(tf.__version__)
cwd = os.getcwd()
print(cwd)
Root_Path=sys.path[0]
print(os.path.isdir("/content/drive/MyDrive"))
if(cwd.startswith("/content")):
  if(os.path.isdir("/content/drive/MyDrive") is False):
    from google.colab import drive
    drive.mount('/content/drive')
  Root_Path = os.path.join(cwd,"/content/drive/MyDrive/keras_segmentatio")
print(Root_Path) 
currentPath = os.path.join(Root_Path,"")
currentdatasetsPath = os.path.join(Root_Path,"dataset/")
checkpoints_path = os.path.join(Root_Path,"resnet50_unet\\")
os.chdir(currentPath)
print(Root_Path) 
print(currentdatasetsPath) 
print(os.path.abspath('.')) # 得到当前文件所在目录的绝对路径
#endregion

plt.figure(num="show",figsize=(10,5)) #设置窗口大小
plt.suptitle('Multi_Image') # 图片名称
realimg = Image.open(os.path.join(Root_Path,"20220629Error/Error/1.bmp"))
plt.subplot(1,3,1), plt.title('real')
plt.imshow(realimg), plt.axis('off')
#annotationsrealimg = Image.open(os.path.join(currentdatasetsPath,"annotations_prepped_test/image (29).bmp"))
#plt.subplot(1,3,2), plt.title('annreal')
#plt.imshow(annotationsrealimg), plt.axis('off')
out = Image.open(os.path.join(Root_Path,"out.png"))
plt.subplot(1,3,3), plt.title('predict')
plt.imshow(out), plt.axis('off')

plt.show()
plt.figure(num="show",figsize=(10,5)) #设置窗口大小
# o = model.predict_segmentation(
#     inp=os.path.join(currentdatasetsPath,"images_prepped_test/0016E5_07965.png"),
#     out_fname=os.path.join(Root_Path,"out2.png") , overlay_img=True, show_legends=True,
#     class_names = [ "Sky",    "Building", "Pole","Road","Pavement","Tree","SignSymbol", "Fence", "Car","Pedestrian", "Bicyclist"]
# )

# plt.imshow(o)
