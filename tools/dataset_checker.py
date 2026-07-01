from pathlib import Path
import cv2
import yaml
from collections import Counter

# ============================================================
# DATASET PATH
# ============================================================

DATASET = Path("/media/cuong/PROJECTS/Dataset")

SETS = [
    ("TRAIN", DATASET / "images" / "train", DATASET / "labels" / "train"),
    ("VAL", DATASET / "images" / "val", DATASET / "labels" / "val"),
]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# Sai số cho phép khi kiểm tra tọa độ
EPS = 1e-6

# ============================================================
# READ DATASET.YAML
# ============================================================

yaml_file = DATASET / "dataset.yaml"

with open(yaml_file, "r") as f:
    cfg = yaml.safe_load(f)

NUM_CLASSES = cfg["nc"]
CLASS_NAMES = cfg["names"]

print("=" * 60)
print("DATASET CONFIG")
print("=" * 60)
print(f"Classes : {NUM_CLASSES}")
print(f"Names   : {CLASS_NAMES}")
print()

# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset(name, image_dir, label_dir):

    print("=" * 60)
    print(name)
    print("=" * 60)

    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(image_dir.glob(f"*{ext}"))

    labels = list(label_dir.glob("*.txt"))

    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")

    missing_labels = []
    missing_images = []
    broken_images = []
    invalid_labels = []

    class_counter = Counter()

    # ---------------------------------------------------------
    # Missing Labels
    # ---------------------------------------------------------

    for img in images:

        txt = label_dir / (img.stem + ".txt")

        if not txt.exists():
            missing_labels.append(img.name)

    # ---------------------------------------------------------
    # Missing Images
    # ---------------------------------------------------------

    for txt in labels:

        found = False

        for ext in IMAGE_EXTENSIONS:

            if (image_dir / (txt.stem + ext)).exists():
                found = True
                break

        if not found:
            missing_images.append(txt.name)

    # ---------------------------------------------------------
    # Broken Images
    # ---------------------------------------------------------

    for img in images:

        im = cv2.imread(str(img))

        if im is None:
            broken_images.append(img.name)

    # ---------------------------------------------------------
    # Validate Labels
    # ---------------------------------------------------------

    for txt in labels:

        try:

            with open(txt, "r") as f:
                lines = f.readlines()

            if len(lines) == 0:
                invalid_labels.append(txt.name)
                continue

            file_valid = True

            for line in lines:

                data = line.strip().split()

                # Polygon phải có:
                # class + ít nhất 4 điểm
                if len(data) < 9:
                    file_valid = False
                    break

                # Sau class phải là các cặp x,y
                if (len(data) - 1) % 2 != 0:
                    file_valid = False
                    break

                cls = int(data[0])

                if cls < 0 or cls >= NUM_CLASSES:
                    file_valid = False
                    break

                class_counter[cls] += 1

                coords = list(map(float, data[1:]))

                for value in coords:

                    # Cho phép sai số floating point
                    if value < -EPS or value > 1 + EPS:
                        file_valid = False
                        break

                if not file_valid:
                    break

            if not file_valid:
                invalid_labels.append(txt.name)

        except Exception:
            invalid_labels.append(txt.name)

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    print()

    print(f"Missing Labels : {len(missing_labels)}")
    print(f"Missing Images : {len(missing_images)}")
    print(f"Broken Images  : {len(broken_images)}")
    print(f"Invalid Labels : {len(invalid_labels)}")

    print()

    print("Class Statistics")
    print("-" * 30)

    for i in range(NUM_CLASSES):

        print(f"{CLASS_NAMES[i]:10s} : {class_counter[i]}")

    print()

    if (
        len(missing_labels) == 0
        and len(missing_images) == 0
        and len(broken_images) == 0
        and len(invalid_labels) == 0
    ):

        print("✅ DATASET OK")

    else:

        print("❌ DATASET HAS ERRORS")

        if len(invalid_labels):

            print("\nInvalid Labels Files")

            for file in invalid_labels:
                print("  -", file)

    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for name, image_dir, label_dir in SETS:

        check_dataset(name, image_dir, label_dir)