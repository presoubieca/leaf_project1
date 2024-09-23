from ultralytics import YOLO
import torch
from PIL import Image
import os


dir = os.getcwd()
model_dir = dir + r"\runs\detect\train_high_Res\weights\best.pt"
model = YOLO(model_dir, task="detect")

for img in os.listdir(dir+r"\\Domati"):
    if img.endswith(".jpg") or img.endswith(".png"):
        path = dir+"\\Domati\\" + img

        results = model.predict(path, save=False, conf=0.5)

        for result in results:
            i = 0
            image = Image.open(path)
            cords = torch.round(result.boxes.xyxy).to(torch.int)
            for x in cords:
                i += 1
                box = image.crop(x.tolist())
                # print(box.size)
                if box.size[0] >= 65 and box.size[1] >= 65:  # may change based on the kind of data and resolution (further testing)
                    if not box.size[0]*2.5 < box.size[1] and not box.size[1]*2.5 < box.size[0]:  # prevents extremely wide but very short or extremely long but very narrow images to appearing
                        box.save("custom_dataset\\Healthy\\"+img[:]+"__"+str(i)+".jpg", verbose=0)

    else:
        print("Wrong format")
