from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="datasets/defect.yaml",
        epochs=5,
        imgsz=640,
        batch=8,
        lr0=0.001,
        name="defect_v5"
    )

    # 使用微调后的模型进行预测
    model = YOLO("runs/detect/defect_v5-4/weights/best.pt")
    results = model.predict(source="static/test.jpg", conf=0.1)
    print("测试结果：")
    print(results[0].boxes.cls)