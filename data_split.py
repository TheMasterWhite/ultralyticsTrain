import os
import shutil
import cv2
import Tools
import torch
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import random


# 将图像批量以数字顺序重命名
def rename_files():
    foldPath = "图像文件夹路径"
    # 文件名起始标签
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
def data_split():
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


# 将图像处理成640x640大小
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


# 删除没有标签的图片
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


# 删除没有图片的标签
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


# 把标签画出来
def draw_labels():
    imagePath = "图像文件夹路径"
    labelPath = "标签文件夹路径"
    outputPath = "输出文件夹路径"
    sum = len(imagePath)
    i = 0
    # 确保输出文件夹存在
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    # 遍历图像文件夹中的所有文件
    for filename in os.listdir(imagePath):
        imageFilePath = os.path.join(imagePath, filename)
        labelFilePath = os.path.join(labelPath, filename.replace(".jpg", ".txt"))
        outputFilePath = os.path.join(outputPath, filename)

        # 读取图像
        i += 1
        print(f"{filename} [{i}/{sum}]")
        image = cv2.imread(imageFilePath)
        image_height, image_width, _ = image.shape

        # 检查标签文件是否存在
        if os.path.exists(labelFilePath):
            # 读取标签文件
            with open(labelFilePath, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    # 解析标签
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    # 将归一化坐标转换为像素值
                    x_center *= image_width
                    y_center *= image_height
                    width *= image_width
                    height *= image_height

                    # 计算边界框的角点坐标
                    x1 = int(x_center - (width / 2))
                    y1 = int(y_center - (height / 2))
                    x2 = int(x_center + (width / 2))
                    y2 = int(y_center + (height / 2))

                    # print(f"{filename} --> x1:{x1} y1:{y1} x2:{x2} y2:{y2}")

                    # 配置不同分类id标注框的颜色
                    if class_id == 0:
                        cv2.rectangle(image, (x1, y2), (x2, y1), (0, 255, 0), 5)
                        cv2.imwrite(outputFilePath, image)
                    elif class_id == 1:
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        cv2.imwrite(output_filepath, image)
                    elif class_id == 2:
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 5)
                        cv2.imwrite(output_filepath, image)


# 检查并输出没有标签的图片名
def check_with_label():
    imagePath = "图像文件夹路径"
    labelPath = "标签文件夹路径"
    for fileName in os.listdir(imagePath):
        rawName = fileName.split(".")[0] + ".txt"
        if not os.path.exists(os.path.join(labelPath, rawName)):
            print(rawName)


# 检查并输出没有图片的标签名
def check_with_image():
    imagePath = "图像文件夹路径"
    labelPath = "标签文件夹路径"
    for fileName in os.listdir(labelPath):
        rawName = fileName.split(".")[0] + ".jpg"
        if not os.path.exists(os.path.join(imagePath, rawName)):
            print(rawName)


if __name__ == '__main__':
    data_split()
