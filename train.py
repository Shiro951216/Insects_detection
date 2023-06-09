from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.yaml")
    model.to('cuda')
    model.train(data="config.yaml", epochs=100, imgsz=480)
