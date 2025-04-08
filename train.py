from ultralytics import YOLO
import os


def main():
    model = YOLO("yolo11l.pt")
    result = model.train(data = "数据集配置文件路径",  # 数据集配置文件路径
                         epochs = 300,  # 迭代轮数，一般不用修改
                         patience = 50,  # 模型训练耐心轮数，一般不用修改
                         batch = 16,  # 批处理大小，较小的batch占用资源少但是速度慢，较大的batch占用资源多速度更快，但是一般在8到64直接修改，最好为2的倍数
                         device = 0,  # 设置训练用的模型id，单卡则为0，多卡为一个列表[0, 1, ...]
                         workers = 24,  # 工作核心数，可以设置成cpu物理核心数量
                         imgsz = 640,  # 图像大小，一般不用修改
                         name = "train")  # 模型输出文件夹名字

    # 训练完毕后自动关机，如果在linux上训练就取消下面的注释
    # os.system("/usr/bin/shutdown")


if __name__ == "__main__":
    main()
