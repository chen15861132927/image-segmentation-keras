#!pip install git+https://github.com/divamgupta/image-segmentation-keras
#!pip install git+https://github.com/divamgupta/image-segmentation-keras
#!pip install git+https://github.com/chenweiSeagate/image-segmentation-keras
#! wget -P /content/drive/MyDrive/keras_segmentatio/dataset https://github.com/divamgupta/datasets/releases/download/seg/dataset1.zip && !unzip /content/drive/MyDrive/keras_segmentatio/dataset1.zip -d  /content/drive/MyDrive/keras_segmentatio/dataset 
#@title head { run: "auto", form-width: "150px" }
#region import
import os
import sys
import random
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
from keras_segmentation.models.unet import resnet50_unet
random.seed(0)
myclass_colors = [(_, _, _) for _ in range(255)]

model =resnet50_unet(n_classes=3 ,  input_height=320, input_width=640  )
EPOCHS =23

def getFileList(dir,Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)
    
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            getFileList(newDir, Filelist, ext)
 
    return Filelist
 
#model.load_weights(os.path.join(checkpoints_path,".00003.index"))
model.train(
    train_images =  os.path.join(currentdatasetsPath,"images_prepped_train/"),
    train_annotations = os.path.join(currentdatasetsPath,"annotations_prepped_train/"),
    val_images=  os.path.join(currentdatasetsPath,"images_prepped_test/"),
    val_annotations= os.path.join(currentdatasetsPath,"annotations_prepped_test/"),
    checkpoints_path =checkpoints_path , epochs=EPOCHS,validate=True,auto_resume_checkpoint=True
)

basePath="C:/Assembly/F20/AVI2/AVIReject CIMBP/0_2/"
resultPath=os.path.join(basePath,"predict/")
if(not os.path.exists(resultPath)):
  os.makedirs(resultPath)

imglist = getFileList(basePath, [], 'bmp')

for imgpath in imglist:
  (path, filename) = os.path.split(imgpath)
  out = model.predict_segmentation(
      inp=os.path.join(Root_Path,imgpath),
      out_fname=os.path.join(resultPath,filename), 
      colors=myclass_colors
  )
#tf.keras.utils.plot_model(model, show_shapes=True)

# plt.figure(num="show",figsize=(10,5)) #设置窗口大小
# plt.suptitle('Multi_Image') # 图片名称
# realimg = Image.open(os.path.join(Root_Path,"20220629Error/Error/1.bmp"))
# plt.subplot(1,3,1), plt.title('real')
# plt.imshow(realimg), plt.axis('off')
# #annotationsrealimg = Image.open(os.path.join(currentdatasetsPath,"annotations_prepped_test/image (29).bmp"))
# #plt.subplot(1,3,2), plt.title('annreal')
# #plt.imshow(annotationsrealimg), plt.axis('off')
# plt.subplot(1,3,3), plt.title('predict')
# plt.imshow(out), plt.axis('off')

# plt.show()
# plt.figure(num="show",figsize=(10,5)) #设置窗口大小
# o = model.predict_segmentation(
#     inp=os.path.join(currentdatasetsPath,"images_prepped_test/0016E5_07965.png"),
#     out_fname=os.path.join(Root_Path,"out2.png") , overlay_img=True, show_legends=True,
#     class_names = [ "Sky",    "Building", "Pole","Road","Pavement","Tree","SignSymbol", "Fence", "Car","Pedestrian", "Bicyclist"]
# )

# plt.imshow(o)
