from ultralytics import YOLO

def main():

    model = YOLO("yolov8n.yaml")
    model.to("cuda")
    results = model.train(data="config.yaml", epochs=110, imgsz=1080, device=0, batch=-1)


if __name__ == "__main__":
    main() 
  