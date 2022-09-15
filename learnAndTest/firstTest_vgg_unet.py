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
plt.ion()
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
checkpoints_path = os.path.join(Root_Path,"vgg_unet\\")
os.chdir(currentPath)
print(Root_Path) 
print(currentdatasetsPath) 
print(os.path.abspath('.')) # 得到当前文件所在目录的绝对路径
#endregion
from keras_segmentation.models.unet import vgg_unet

model =vgg_unet(n_classes=12 ,  input_height=320, input_width=640  )
EPOCHS =10

#model.load_weights(os.path.join(checkpoints_path,".00003.index"))
model.train(
    train_images =  os.path.join(currentdatasetsPath,"images_prepped_train/"),
    train_annotations = os.path.join(currentdatasetsPath,"annotations_prepped_train/"),
    val_images=  os.path.join(currentdatasetsPath,"images_prepped_test/"),
    val_annotations= os.path.join(currentdatasetsPath,"annotations_prepped_test/"),
    checkpoints_path =checkpoints_path , epochs=EPOCHS,validate=True,auto_resume_checkpoint=True
)

out = model.predict_segmentation(
    inp=os.path.join(currentdatasetsPath,"images_prepped_test/0016E5_07965.png"),
    out_fname=os.path.join(Root_Path,"out.png")
)

plt.imshow(out)



o = model.predict_segmentation(
    inp=os.path.join(currentdatasetsPath,"images_prepped_test/0016E5_07965.png"),
    out_fname=os.path.join(Root_Path,"out2.png") , overlay_img=True, show_legends=True,
    class_names = [ "Sky",    "Building", "Pole","Road","Pavement","Tree","SignSymbol", "Fence", "Car","Pedestrian", "Bicyclist"]
)

plt.imshow(o)
