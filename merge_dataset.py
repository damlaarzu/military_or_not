import os
import shutil
import random


military_base = "military-vehicle.v6i.yolov11"


nwpu_images_dir = "nwpu vhr10 dataset/positive image set"
nwpu_labels_dir = "nwpu vhr10 dataset/labels_yolo"


output_base = "datasets/merged"
output_images = os.path.join(output_base, "images")
output_labels = os.path.join(output_base, "labels")

# split ratios for nwpu
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1



os.makedirs(output_base, exist_ok=True)
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_images, split), exist_ok=True)
    os.makedirs(os.path.join(output_labels, split), exist_ok=True)


#copying military dataset
for split in ["train", "val", "test"]:
    img_dir = os.path.join(military_base, split, "images")
    lbl_dir = os.path.join(military_base, split, "labels")

    for file in os.listdir(lbl_dir):
        if not file.endswith(".txt"):
            continue
        img_path = os.path.join(img_dir, file.replace(".txt", ".jpg"))
        lbl_path = os.path.join(lbl_dir, file)

        if not os.path.exists(img_path):
            continue

        out_img_path = os.path.join(output_images, split, os.path.basename(img_path))
        out_lbl_path = os.path.join(output_labels, split, os.path.basename(lbl_path))

        shutil.copy(img_path, out_img_path)
        shutil.copy(lbl_path, out_lbl_path)

print(" Military dataset merged!")

# ===================== NWPU Dataset Split and Copying =====================

all_nwpu = []
for file in os.listdir(nwpu_labels_dir):
    if not file.endswith(".txt"):
        continue
    label_path = os.path.join(nwpu_labels_dir, file)
    image_path = os.path.join(nwpu_images_dir, file.replace(".txt", ".jpg"))
    if os.path.exists(image_path):
        all_nwpu.append((image_path, label_path))

random.shuffle(all_nwpu)
n = len(all_nwpu)
train_end = int(n * train_ratio)
val_end = train_end + int(n * val_ratio)

splits = {
    "train": all_nwpu[:train_end],
    "val": all_nwpu[train_end:val_end],
    "test": all_nwpu[val_end:]
}

for split, items in splits.items():
    for img_path, lbl_path in items:
        out_img_path = os.path.join(output_images, split, os.path.basename(img_path))
        out_lbl_path = os.path.join(output_labels, split, os.path.basename(lbl_path))

        shutil.copy(img_path, out_img_path)
        shutil.copy(lbl_path, out_lbl_path)

print("Completed NWPU dataset merge & split!")
