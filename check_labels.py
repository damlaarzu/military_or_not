import os
from collections import Counter

def count_labels(label_dir):
    counter = Counter()
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            with open(os.path.join(label_dir, file), "r") as f:
                for line in f.readlines():
                    if line.strip():  
                        cls_id = int(line.split()[0])
                        counter[cls_id] += 1
    return counter

def main():
    train_labels = "data/labels/train"
    val_labels = "data/labels/val"

    print(" Label Distribution\n" + "="*30)

    if os.path.exists(train_labels):
        train_counts = count_labels(train_labels)
        print("Train set:", dict(train_counts))
    else:
        print(" Train labels klasörü bulunamadı")

    if os.path.exists(val_labels):
        val_counts = count_labels(val_labels)
        print("Val set:", dict(val_counts))
    else:
        print(" Val labels klasörü bulunamadı")

if __name__ == "__main__":
    main()
