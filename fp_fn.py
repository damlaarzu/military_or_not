import cv2
import random
from ultralytics import YOLO

model = YOLO("runs/train/military_or_not/weights/best.pt")


val_images = "data/images/val"

# any image in the validation set
results = model.predict(source=val_images, save=True, conf=0.25)


