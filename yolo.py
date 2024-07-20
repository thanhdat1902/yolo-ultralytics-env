from ultralytics import YOLO
# Display model information (optional)
if __name__ == '__main__':
    # Build a YOLOv9c model from scratch
    model = YOLO("yolov10x.yaml")


    # Build a YOLOv9c model from pretrained weight
    model = YOLO("yolov10x.pt")
    model.info()

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="./data_mobile.yaml", epochs=120, imgsz=640, batch=8)

    # Run inference with the YOLOv9c model on the 'bus.jpg' image
    results = model("image.png")
