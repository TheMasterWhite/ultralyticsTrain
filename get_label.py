from ultralytics import YOLO
import os, json, shutil
from torchvision.transforms import ToTensor, Resize
from concurrent.futures import ThreadPoolExecutor
import cv2
from PIL import Image
import torch
import numpy as np


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


def resize_image(ImagePath):
    lst = [i for i in os.listdir(ImagePath)]
    count = len(os.listdir(ImagePath))
    cnt = 0
    resize_transform = Resize((640, 640))
    for fileName in lst:
        cnt += 1
        imageObj = Image.open(os.path.join(ImagePath, fileName))
        image_tensor = ToTensor()(imageObj)
        resized_image_tensor = resize_transform(image_tensor)
        resized_image = Image.fromarray(resized_image_tensor.mul(255).permute(1, 2, 0).byte().numpy())
        saveImgPath = os.path.join(ImagePath, fileName)
        resized_image.save(saveImgPath)
        print(f"Resize: {fileName} [{cnt}/{count}]")


def get_label(Result, ImagePath):
    lines = []
    for img in Result:
        name = img["name"]
        box = img["box"]
        className = 0
        # if name == "regulator":
        #     className = 0
        # else:
        #     className = 1

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
        line = f"{className} {center_x_normalized:.6f} {center_y_normalized:.6f} {normalizedWidth:.6f} {normalizedHeight:.6f}"
        lines.append(line)
    return lines


def main():
    imageFolder = "图像文件夹路径"
    labelFolder = "标签文件夹路径"
    if not os.path.exists(labelFolder):
        os.makedirs(labelFolder)
    resize_image(imageFolder)
    model = YOLO(model = "模型文件夹", task = "detect")
    lst = [i for i in os.listdir(imageFolder)]


    def detect(fileName):
        imagePath = os.path.join(imageFolder,
                                 fileName)
        result = model(source = imagePath, save = False, conf = 0.6, imgsz = 640)
        detectResult = json.loads(result[0].to_json())
        if detectResult == []:
            os.remove(imagePath)
            return
        labelList = get_label(detectResult, imagePath)
        labelPath = imagePath.replace("images", "labels")
        labelPath = labelPath.replace("jpg", "txt")
        with open(labelPath, "w") as f:
            for label in labelList:
                f.write(label + "\n")


    with ThreadPoolExecutor(max_workers = 24) as executor:
        executor.map(detect, lst)

    delete_images(imageFolder, labelFolder)
    random_contrast(imageFolder, labelFolder)


if __name__ == "__main__":
    main()
