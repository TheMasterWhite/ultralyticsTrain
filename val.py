from ultralytics import YOLO


def main():
    model = YOLO("训练好的模型路径")
    model.val(data = "数据集配置文件路径",
              split = "test",  # 划分类型
              imgsz = 640,
              batch = 1,  # 批次大小，一般不用修改
              project = "runs/val",
              name = "exp",
              )


if __name__ == "__main__":
    main()
