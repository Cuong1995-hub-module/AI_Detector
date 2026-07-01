from pathlib import Path
import cv2
import numpy as np
import yaml
import random

# ============================================================
# DATASET PATH
# ============================================================

DATASET = Path("/media/cuong/PROJECTS/Dataset")

IMAGE_DIR = DATASET / "images" / "train"
LABEL_DIR = DATASET / "labels" / "train"

SAVE_DIR = Path("preview")
SAVE_DIR.mkdir(exist_ok=True)

# ============================================================
# READ DATASET YAML
# ============================================================

with open(DATASET / "dataset.yaml", "r") as f:
    cfg = yaml.safe_load(f)

CLASS_NAMES = cfg["names"]

# ============================================================
# IMAGE LIST
# ============================================================

IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

images = []

for ext in IMAGE_EXTENSIONS:
    images.extend(sorted(IMAGE_DIR.glob(ext)))

print("=" * 60)
print(f"Found {len(images)} images")
print("=" * 60)

# ============================================================
# START
# ============================================================

index = 0

while True:

    img_path = images[index]

    label_path = LABEL_DIR / (img_path.stem + ".txt")

    image = cv2.imread(str(img_path))

    if image is None:

        print("Cannot read:", img_path.name)

        index += 1
        continue

    H, W = image.shape[:2]

    object_count = 0

    # --------------------------------------------------------

    if label_path.exists():

        with open(label_path, "r") as f:

            lines = f.readlines()

        for line in lines:

            data = line.strip().split()

            if len(data) < 9:
                continue

            cls = int(data[0])

            coords = list(map(float, data[1:]))

            pts = []

            for i in range(0, len(coords), 2):

                x = int(coords[i] * W)
                y = int(coords[i + 1] * H)

                pts.append([x, y])

            pts = np.array(pts, np.int32)

            # Draw polygon

            cv2.polylines(
                image,
                [pts],
                True,
                (0, 255, 0),
                2,
            )

            # Draw corner

            for p in pts:

                cv2.circle(
                    image,
                    tuple(p),
                    4,
                    (0, 0, 255),
                    -1,
                )

            # Draw class

            cv2.putText(
                image,
                CLASS_NAMES[cls],
                tuple(pts[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            object_count += 1

    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------

    cv2.putText(
        image,
        f"Image : {index+1}/{len(images)}",
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2,
    )

    cv2.putText(
        image,
        img_path.name,
        (20,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2,
    )

    cv2.putText(
        image,
        f"Resolution : {W} x {H}",
        (20,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2,
    )

    cv2.putText(
        image,
        f"Objects : {object_count}",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2,
    )

    # --------------------------------------------------------
    # HELP
    # --------------------------------------------------------

    cv2.putText(
        image,
        "N:Next  P:Prev  R:Random  F:First  L:Last  S:Save  Q:Quit",
        (20,H-20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0,255,255),
        2,
    )

    cv2.imshow("Dataset Visualizer V3", image)

    key = cv2.waitKey(0) & 0xFF

    # Quit

    if key == ord('q') or key == 27:
        break

    # Next

    elif key == ord('n'):

        index = min(index+1,len(images)-1)

    # Previous

    elif key == ord('p'):

        index = max(index-1,0)

    # Random

    elif key == ord('r'):

        index = random.randint(0,len(images)-1)

    # First

    elif key == ord('f'):

        index = 0

    # Last

    elif key == ord('l'):

        index = len(images)-1

    # Save

    elif key == ord('s'):

        save_path = SAVE_DIR / img_path.name

        cv2.imwrite(str(save_path),image)

        print("Saved:",save_path)

cv2.destroyAllWindows()