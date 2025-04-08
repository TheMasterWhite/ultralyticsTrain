import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import hflip
from torchvision.transforms.functional import adjust_contrast
from PIL import Image


def random_contrast():
    imagePath = "图像文件夹路径"
    labelPath = "标签文件夹路径"
    saveImagePath = imagePath
    saveLabelPath = labelPath
    count = len(os.listdir(imagePath))
    cnt = 0

    for fullFileName in [i for i in os.listdir(imagePath)]:
        cnt += 1
        imageObj = Image.open(os.path.join(imagePath, fullFileName)).convert('RGB')
        imageTensor = torch.tensor(np.array(imageObj), dtype = torch.float32).permute(2, 0, 1) / 255.0
        mean = imageTensor.mean(dim = [1, 2], keepdim = True)
        adjustedImageTensor = torch.clamp((imageTensor - mean) * 1.3 + mean, 0.0, 1.0)
        adjustedImage = Image.fromarray((adjustedImageTensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        adjustedImage.save(os.path.join(saveImagePath, "contrast_" + fullFileName))

        txtName = fullFileName.split(".")[0] + ".txt"
        shutil.copy(os.path.join(labelPath, txtName), os.path.join(saveLabelPath, "contrast_" + txtName))
        print(f"{fullFileName} [{cnt}/{count}]")


if __name__ == '__main__':
    random_contrast()
