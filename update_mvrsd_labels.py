import os

def update_mvrsd_labels():
    mvrsd_labels_dirs = ['mvrsd/labels/train', 'mvrsd/labels/val']

    # New class ID (1: military_vehicle)
    NEW_CLASS_ID = '1'

    for labels_path in mvrsd_labels_dirs:
        # List label files
        label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
        
        for file_name in label_files:
            file_path = os.path.join(labels_path, file_name)
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    # Change class ID (1: military_vehicle)
                    parts[0] = NEW_CLASS_ID
                    new_line = ' '.join(parts) + '\n'
                    new_lines.append(new_line)

            # Write updated labels back to file
            with open(file_path, 'w') as f:
                f.writelines(new_lines)

    print("MVRSD labels successfully updated to military vehicle (class 1).")

if __name__ == '__main__':
    update_mvrsd_labels()