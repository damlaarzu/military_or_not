from ultralytics import YOLO

def main():

    model = YOLO("runs/train/military_or_not/weights/best.pt")

    # Validation 
    metrics = model.val(save_json=True, save_conf=True, plots=True)

    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

if __name__ == "__main__":
    main()
