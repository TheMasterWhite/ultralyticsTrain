import os
import shutil
import cv2
import Tools
import torch
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import random


# 将图像重命名
def rename_files():
    foldPath = "图像文件夹路径"
    # 文件名其实标签
    nameIndex = 1
    for fileName in os.listdir(foldPath):
        newName = str(nameIndex) + ".jpg"
        os.rename(os.path.join(foldPath, fileName), os.path.join(foldPath, newName))
        nameIndex += 1
        print(nameIndex)


# 创建输出数据集的文件夹结构
def create_folder(dataset_root):
    folders = ["train", "val", "test"]
    for folder in folders:
        os.makedirs(os.path.join(dataset_root, folder, "images"), exist_ok = True)
        os.makedirs(os.path.join(dataset_root, folder, "labels"), exist_ok = True)


# 划分数据集
def DataSplit():
    imageFolderPath = "图像文件夹路径"
    labelFolderPath = "标签文件夹路径"
    outputFolderPath = "输出文件夹路径"

    if not os.path.exists(imageFolderPath):
        os.makedirs(imageFolderPath)

    fullFileNames = [i for i in os.listdir(imageFolderPath)]
    nums = len(fullFileNames)
    index = 0
    # 创建文件夹结构
    create_folder(outputFolderPath)

    if nums < 1000:
        Scale = 0.8
        random.shuffle(fullFileNames)

        valImageNum = int(nums * (1 - Scale))
        trainImageNum = nums - valImageNum

        # 划分训练集和验证集
        for i in range(trainImageNum):
            fullFileName = fullFileNames[i]
            fileName = os.path.splitext(fullFileName)[0]
            txtFileName = fileName + ".txt"
            shutil.copy(os.path.join(imageFolderPath, fullFileName),
                        os.path.join(outputFolderPath, 'train', 'images', fullFileName))
            index += 1
            print(f"Copied {fullFileName} [{index} / {nums}]")
            # 如果存在标签就复制标签
            if os.path.exists(os.path.join(labelFolderPath, txtFileName)):
                shutil.copy(os.path.join(labelFolderPath, txtFileName),
                            os.path.join(outputFolderPath, 'train', 'labels', txtFileName))

        for i in range(trainImageNum, trainImageNum + valImageNum):
            fullFileName = fullFileNames[i]
            fileName = os.path.splitext(fullFileName)[0]
            txtFileName = fileName + ".txt"
            shutil.copy(os.path.join(imageFolderPath, fullFileName),
                        os.path.join(outputFolderPath, 'val', 'images', fullFileName))
            index += 1
            print(f"Copied {fullFileName} [{index} / {nums}]")
            # 如果存在标签就复制标签
            if os.path.exists(os.path.join(labelFolderPath, txtFileName)):
                shutil.copy(os.path.join(labelFolderPath, txtFileName),
                            os.path.join(outputFolderPath, 'val', 'labels', txtFileName))
    else:
        Scale_train = 0.7  # 训练集比例为70%
        Scale_val = 0.2  # 验证集比例为20%

        random.shuffle(fullFileNames)

        trainImageNum = int(nums * Scale_train)
        valImageNum = int(nums * Scale_val)
        testImageNum = nums - trainImageNum - valImageNum

        # 划分训练集、验证集和测试集
        for i in range(trainImageNum):
            fullFileName = fullFileNames[i]
            fileName = os.path.splitext(fullFileName)[0]
            txtFileName = fileName + ".txt"
            shutil.copy(os.path.join(imageFolderPath, fullFileName),
                        os.path.join(outputFolderPath, "train", "images", fullFileName))
            index += 1
            print(f"Copied {fullFileName} [{index} / {nums}]")
            # 如果存在标签就复制标签
            if os.path.exists(os.path.join(labelFolderPath, txtFileName)):
                shutil.copy(os.path.join(labelFolderPath, txtFileName),
                            os.path.join(outputFolderPath, "train", "labels", txtFileName))

        for i in range(trainImageNum, trainImageNum + valImageNum):
            fullFileName = fullFileNames[i]
            fileName = os.path.splitext(fullFileName)[0]
            txtFileName = fileName + ".txt"
            shutil.copy(os.path.join(imageFolderPath, fullFileName),
                        os.path.join(outputFolderPath, "val", "images", fullFileName))
            index += 1
            print(f"Copied {fullFileName} [{index} / {nums}]")
            # 如果存在标签就复制标签
            if os.path.exists(os.path.join(labelFolderPath, txtFileName)):
                shutil.copy(os.path.join(labelFolderPath, txtFileName),
                            os.path.join(outputFolderPath, "val", "labels", txtFileName))

        for i in range(trainImageNum + valImageNum, trainImageNum + valImageNum + testImageNum):
            fullFileName = fullFileNames[i]
            fileName = os.path.splitext(fullFileName)[0]
            txtFileName = fileName + ".txt"
            shutil.copy(os.path.join(imageFolderPath, fullFileName),
                        os.path.join(outputFolderPath, "test", "images", fullFileName))
            index += 1
            print(f"Copied {fullFileName} [{index} / {nums}]")
            # 如果存在标签就复制标签
            if os.path.exists(os.path.join(labelFolderPath, txtFileName)):
                shutil.copy(os.path.join(labelFolderPath, txtFileName),
                            os.path.join(outputFolderPath, "test", "labels", txtFileName))
    print("数据集划分完毕")


# 将图像处理成640x640大学
def resize_image():
    imagePath = "原始图像文件夹路径"
    savePath = "输出图像文件夹路径"
    lst = [i for i in os.listdir(imagePath)]
    count = len(os.listdir(imagePath))
    cnt = 0
    resize_transform = Resize((640, 640))
    for fileName in lst:
        cnt += 1
        imageObj = Image.open(os.path.join(imagePath, fileName))
        image_tensor = ToTensor()(imageObj)
        resized_image_tensor = resize_transform(image_tensor)
        resized_image = Image.fromarray(resized_image_tensor.mul(255).permute(1, 2, 0).byte().numpy())
        saveImgPath = os.path.join(savePath, fileName)
        resized_image.save(saveImgPath)
        print(f"{fileName} [{cnt}/{count}]")


# 删除没有label的图片
def delete_images():
    imagePath = "图像文件夹路径"
    labelPath = "标签文件夹路径"
    labelList = os.listdir(labelPath)
    for fullFileName in os.listdir(imagePath):
        fileName = Tools.GetFileName(fullFileName)
        txtName = fileName + ".txt"
        if txtName not in labelList:
            os.remove(os.path.join(imagePath, fullFileName))
            print(fullFileName)


# 删除没有图片的label
def delete_labels():
    imagePath = "图像文件夹路径"
    labelPath = "标签文件夹路径"
    imageList = os.listdir(imagePath)
    for fullFileName in os.listdir(labelPath):
        fileName = fullFileName.split(".")[0]
        imageName = fileName + ".jpg"
        if imageName not in imageList:
            os.remove(os.path.join(labelPath, fullFileName))
            print(fullFileName)


if __name__ == '__main__':
    DataSplit()
