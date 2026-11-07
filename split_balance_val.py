import os
import shutil
import random

image_dir = "data/images"
label_dir = "data/labels"

train_images = os.path.join(image_dir, "train")
train_labels = os.path.join(label_dir, "train")
val_images = os.path.join(image_dir, "val")
val_labels = os.path.join(label_dir, "val")

# if val folders don't exist, create them
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# Group files by Class ID
def get_class_files(label_folder):
    class_files = {}
    for file in os.listdir(label_folder):
        if file.endswith(".txt"):
            with open(os.path.join(label_folder, file), "r") as f:
                lines = f.readlines()
                for line in lines:
                    cls_id = int(line.split()[0])
                    if cls_id not in class_files:
                        class_files[cls_id] = []
                    class_files[cls_id].append(file)
                    break
    return class_files

class_files = get_class_files(train_labels)

# For each class, select 20% for validation and move
for cls_id, files in class_files.items():
    val_count = int(len(files) * 0.2)
    val_selected = random.sample(files, val_count)
    print(f"Class {cls_id}: {len(files)} files, {val_count} selected for validation.")

    for file in val_selected:
        # Label file
        label_src = os.path.join(train_labels, file)
        label_dst = os.path.join(val_labels, file)
        shutil.move(label_src, label_dst)

        # Image file
        img_file_jpg = file.replace(".txt", ".jpg")
        img_file_png = file.replace(".txt", ".png")
        img_src_jpg = os.path.join(train_images, img_file_jpg)
        img_src_png = os.path.join(train_images, img_file_png)

        if os.path.exists(img_src_jpg):
            shutil.move(img_src_jpg, os.path.join(val_images, img_file_jpg))
        elif os.path.exists(img_src_png):
            shutil.move(img_src_png, os.path.join(val_images, img_file_png))
