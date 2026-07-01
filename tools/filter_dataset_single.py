from pathlib import Path
import shutil

# ============================================================
# PATH
# ============================================================

SOURCE = Path("/media/cuong/PROJECTS/Dataset")
TARGET = Path("/media/cuong/PROJECTS/Dataset_Single")

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

SETS = [
    "train",
    "val",
]

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

for subset in SETS:

    (TARGET / "images" / subset).mkdir(
        parents=True,
        exist_ok=True,
    )

    (TARGET / "labels" / subset).mkdir(
        parents=True,
        exist_ok=True,
    )

# ============================================================
# COPY DATASET.YAML
# ============================================================

yaml_file = SOURCE / "dataset.yaml"

if yaml_file.exists():

    shutil.copy2(
        yaml_file,
        TARGET / "dataset.yaml",
    )

# ============================================================
# STATISTICS
# ============================================================

total_images = 0

kept_images = 0

removed_images = 0

removed_multi_object = 0

removed_invalid_polygon = 0

print("=" * 60)
print("FILTER DATASET")
print("=" * 60)

# ============================================================
# FILTER
# ============================================================

for subset in SETS:

    image_dir = SOURCE / "images" / subset

    label_dir = SOURCE / "labels" / subset

    images = []

    for ext in IMAGE_EXTS:

        images.extend(image_dir.glob(f"*{ext}"))

    images = sorted(images)

    keep = 0

    remove = 0

    print()
    print(subset.upper())
    print("-" * 30)

    for img in images:

        total_images += 1

        label = label_dir / (img.stem + ".txt")

        # ----------------------------------------
        # Label không tồn tại
        # ----------------------------------------

        if not label.exists():

            remove += 1
            removed_images += 1

            continue

        with open(label, "r") as f:

            lines = [
                x.strip()
                for x in f.readlines()
                if x.strip()
            ]

        # ----------------------------------------
        # Chỉ giữ đúng 1 object
        # ----------------------------------------

        if len(lines) != 1:

            remove += 1
            removed_images += 1
            removed_multi_object += 1

            continue

        data = lines[0].split()

        # ----------------------------------------
        # class + 8 tọa độ = 9 phần tử
        # ----------------------------------------

        if len(data) != 9:

            remove += 1
            removed_images += 1
            removed_invalid_polygon += 1

            continue

        # ----------------------------------------
        # Kiểm tra class
        # ----------------------------------------

        try:

            cls = int(data[0])

        except:

            remove += 1
            removed_images += 1
            removed_invalid_polygon += 1

            continue

        # ----------------------------------------
        # Kiểm tra tọa độ
        # ----------------------------------------

        try:

            coords = list(map(float, data[1:]))

        except:

            remove += 1
            removed_images += 1
            removed_invalid_polygon += 1

            continue

        valid = True

        for value in coords:

            if value < 0 or value > 1:

                valid = False
                break

        if not valid:

            remove += 1
            removed_images += 1
            removed_invalid_polygon += 1

            continue

        # ----------------------------------------
        # COPY
        # ----------------------------------------

        shutil.copy2(
            img,
            TARGET / "images" / subset / img.name,
        )

        shutil.copy2(
            label,
            TARGET / "labels" / subset / label.name,
        )

        keep += 1
        kept_images += 1

    print(f"Keep    : {keep}")
    print(f"Removed : {remove}")

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Total Images                 : {total_images}")
print(f"Kept Images                  : {kept_images}")
print(f"Removed Images               : {removed_images}")
print(f"Multiple Objects Removed     : {removed_multi_object}")
print(f"Polygon != 4 Points Removed  : {removed_invalid_polygon}")

if total_images > 0:

    remain = kept_images / total_images * 100

    print(f"Remain Ratio                 : {remain:.2f}%")

print()
print("Done.")
print("Output :", TARGET)