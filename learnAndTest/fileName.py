import os
basePath="C:/Dev/GithubProject/image-segmentation-keras/AVIDLProject/20220629Error/Error/"

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
 
imglist = getFileList(basePath, [], 'bmp')

for imgpath in imglist:
    (path, filename) = os.path.split(imgpath)
    print(filename)
    print(path)
# (file, ext) = os.path.splitext(url)
# print(file)
# print(ext)

# (path, filename) = os.path.split(url)
# print(filename)
# print(path)
