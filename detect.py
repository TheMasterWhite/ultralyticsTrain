from ultralytics import YOLO
import os
import shutil
import json
import torch
from concurrent.futures import ThreadPoolExecutor


def detect(Model, Source):
    model = YOLO(model = Model, task = "detect")
    result = model(source = Source, save = True, conf = 0.6)
    detectResult = json.loads(result[0].to_json())
    print(detectResult)


# 多线程搜索存在目标物体的图片
def search():
    imagePath = "图像文件夹路径"
    savePath = "输出图像文件夹路径"
    modelConf = "模型置信度，默认0.5"
    modelPath = "模型路径"
    lst = [i for i in os.listdir(imagePath)]
    retList = []
    num = 0
    # 加载模型
    buckleModel = YOLO(model = modelPath,
                       task = "detect")


    def process_image(fileName):
        path = imagePath + "/" + fileName
        result = buckleModel(source = path, save = False, conf = modelConf)
        detectResult = json.loads(result[0].to_json())

        if detectResult != []:
            for info in detectResult:
                shutil.copy(path, savePath + "/" + fileName)


    # 使用线程池执行图像处理
    with ThreadPoolExecutor(max_workers = 24) as executor:
        executor.map(process_image, lst[::-1])


if __name__ == '__main__':
    modelPath = "模型路径"
    source = "待检测图片路径（文件或文件夹路径）"
    detect(modelPath, source)
    # search()
