from ultralytics import YOLO
import os, json, shutil
from torchvision.transforms import ToTensor, Resize
from concurrent.futures import ThreadPoolExecutor
import cv2
from PIL import Image
import torch
import numpy as np


# 随机对比度
def random_contrast(ImagePath, LabelPath):
    count = len(os.listdir(ImagePath))
    cnt = 0

    for fullFileName in [i for i in os.listdir(ImagePath)]:
        cnt += 1
        imageObj = Image.open(os.path.join(ImagePath, fullFileName)).convert("RGB")
        imageTensor = torch.tensor(np.array(imageObj), dtype = torch.float32).permute(2, 0, 1) / 255.0
        mean = imageTensor.mean(dim = [1, 2], keepdim = True)
        adjustedImageTensor = torch.clamp((imageTensor - mean) * 1.3 + mean, 0.0, 1.0)
        adjustedImage = Image.fromarray((adjustedImageTensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        adjustedImage.save(os.path.join(ImagePath, "contrast_" + fullFileName))

        txtName = fullFileName.split(".")[0] + ".txt"
        shutil.copy(os.path.join(LabelPath, txtName), os.path.join(LabelPath, "contrast_" + txtName))
        print(f"Contrast: {fullFileName} [{cnt}/{count}]")


# 删除没有label的图片
def delete_images(ImagePath, LabelPath):
    labelList = os.listdir(LabelPath)
    num = 0
    for fullFileName in os.listdir(ImagePath):
        fileName, extension = os.path.splitext(fullFileName)
        txtName = fileName + ".txt"
        if txtName not in labelList:
            os.remove(os.path.join(ImagePath, fullFileName))
            num += 1
            print(f"Deleted no label image: {fullFileName}")
    print(f"Deleted {num} images")


# 获取多类别标签
def get_label_mutileclass(Result, ImagePath):
    try:
        lines = []
        for result in Result:
            # 类型名
            className = result["name"]
            if className == "sign":
                classId = 0
            elif className == "arrow":
                classId = 1
            elif className == "valve":
                classId = 2
            # elif className == "regulator":
            #     classId = 3
            # elif className == "regulatorButton":
            #     classId = 4
            else:
                continue

            # 标注框
            box = result["box"]
            image = cv2.imread(ImagePath)
            imageHeight, imageWidth, _ = image.shape

            x1 = box["x1"]
            x2 = box["x2"]
            y1 = box["y1"]
            y2 = box["y2"]

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            # 归一化中心点坐标
            center_x_normalized = center_x / imageWidth
            center_y_normalized = center_y / imageHeight
            # 计算标注框规格
            new_width = x2 - x1
            new_height = y2 - y1
            # 计算归一化标注框数据
            normalizedWidth = new_width / imageHeight
            normalizedHeight = new_height / imageWidth
            line = f"{classId} {center_x_normalized:.6f} {center_y_normalized:.6f} {normalizedWidth:.6f} {normalizedHeight:.6f}"
            lines.append(line)
        return lines
    except Exception as e:
        print(e)
        raise


# 自动化图像预处理与打标主程序
def main(modelPath, ImageFolder, LabelFolder):
    try:
        modelConf = 0.7
        if not os.path.exists(LabelFolder):
            os.makedirs(LabelFolder)
        # resize_image(imageFolder)
        model = YOLO(model = modelPath, task = "detect")
        lst = [i for i in os.listdir(ImageFolder)]


        def detect(fileName):
            try:
                imagePath = os.path.join(ImageFolder,
                                         fileName)
                imageObj = cv2.imread(imagePath)
                height, width = imageObj.shape[:2]
                size = (height, width)
                result = model(source = imagePath, save = False, conf = modelConf, imgsz = size)
                detectResult = json.loads(result[0].to_json())
                print(detectResult)
                if detectResult == []:
                    os.remove(imagePath)
                    return
                labelList = get_label_mutileclass(detectResult, imagePath)
                print(labelList)
                labelPath = imagePath.replace("images", "labels")
                labelPath = labelPath.replace("jpg", "txt")
                with open(labelPath, "a") as f:
                    for label in labelList:
                        f.write(label + "\n")
            except Exception as e:
                print(e)
                raise


        with ThreadPoolExecutor(max_workers = 24) as executor:
            executor.map(detect, lst)

    except Exception as e:
        print(e)
        raise


# 如果要同时使用多个模型对同一批图片进行打标，请使用此函数，否则只执行main即可
def process():
    modelPath = "模型路径"
    imageFolder = "数据集路径"
    labelFolder = "标签保存路径"
    main(modelPath)
    # 删除没有label的图片
    # delete_images(imageFolder, labelFolder)
    # 随机对比度
    # random_contrast(imageFolder, labelFolder)


if __name__ == "__main__":
    # 使用前请在get_label_mutileclass方法中定义类别名称
    imageFolder = "数据集路径"
    labelFolder = "标签保存路径"
    main("F:/Download/best.pt", imageFolder, labelFolder)
