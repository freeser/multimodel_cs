from ultralytics import YOLO
import numpy as np
from config import YOLO_MODEL

# model = YOLO(r"D:\Study\python\multimodel_cs\yolo11n.pt")
model = YOLO(YOLO_MODEL)

def detect_image(image: np.ndarray):
    results = model(image)

    # 提取检测结果
    detections = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])    # 置信度
            class_id = int(box.cls[0])         # 类别ID
            class_name = model.names[class_id]  # 类别名称
            detections.append(f"{class_name} ({confidence:.2f})")
    if detections:
        return "；".join(detections)
    return "未检测到明显商品异常"


# 测试代码
# if __name__ == "__main__":
#     import cv2
#     image = cv2.imread(r"D:\Study\python\multimodel_cs\static\test.jpg")
#     print(detect_image(image))
