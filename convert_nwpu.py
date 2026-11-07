import os
from PIL import Image


labels_dir = "nwpu vhr10 dataset/ground truth"
images_dir = "nwpu vhr10 dataset/positive image set"
output_labels_dir = "nwpu vhr10 dataset/labels_yolo"
os.makedirs(output_labels_dir, exist_ok=True)


target_classes = {1, 2, 10}  # airplane, ship, vehicle

for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    image_file = label_file.replace(".txt", ".jpg")
    img_path = os.path.join(images_dir, image_file)
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path)
    img_w, img_h = img.size

    with open(os.path.join(labels_dir, label_file), "r") as f:
        lines = f.readlines()

    yolo_lines = []
    for line in lines:
        # Format: (x1,y1),(x2,y2),a
        line = line.strip().replace("(", "").replace(")", "")
        parts = line.split(",")
        if len(parts) < 5:
            continue

        xmin, ymin, xmax, ymax, cls_id = map(int, parts)
        
        if cls_id not in target_classes:
            continue  # pass other classes

        # All of them will be civilian_vehicle
        new_cls = 1

        # YOLO format
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        yolo_lines.append(f"{new_cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    if yolo_lines:
        with open(os.path.join(output_labels_dir, label_file), "w") as f:
            f.writelines(yolo_lines)
