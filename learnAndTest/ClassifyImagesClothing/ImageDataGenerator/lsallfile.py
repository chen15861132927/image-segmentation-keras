import os
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

cwd = os.getcwd()
print(cwd)
Root_Path=sys.path[0]
googlePath="/content/drive/MyDrive/"
print(os.path.isdir(googlePath))
if(cwd.startswith("/content")):
  if(os.path.isdir(googlePath) is False):
    from google.colab import drive
    drive.mount('/content/drive')
  Root_Path = os.path.join(cwd,googlePath)
print(Root_Path) 
Root_Path=r'C:\Dev\GithubProject\dlwithtf\ClassifyImagesClothing\LearnResnetMnistAsClass'
path=os.path.join(Root_Path,"MnistDatagenTrain")

allfile=os.listdir(path)
def ShowAllFile(FoldPath):
    allfile=os.listdir(FoldPath)
    itemcount=0
    for fileitem in allfile:
        tempDirPath=os.path.join(FoldPath,fileitem)
        if os.path.isdir(tempDirPath):
            #print(tempDirPath,"it's a directory")
            tempPath=os.path.join(FoldPath,fileitem)
            ShowAllFile(tempPath)
        elif os.path.isfile(tempDirPath):
            #if tempDirPath.find("/FlowPath/")>0:
            #os.remove(tempDirPath)
            #print(tempDirPath," it's a normal file")
            #print(str(itemcount))
            itemcount=itemcount+1 
    
    print(FoldPath, itemcount)

ShowAllFile(path)