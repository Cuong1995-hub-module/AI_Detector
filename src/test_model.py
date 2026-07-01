import torch

from dataset import PlateDataset
from torch.utils.data import DataLoader
from model import PlateDetector

# ============================================================
# PATH
# ============================================================

IMAGE_DIR = "/media/cuong/PROJECTS/Dataset_Single/images/train"
LABEL_DIR = "/media/cuong/PROJECTS/Dataset_Single/labels/train"

# ============================================================
# DATASET
# ============================================================

dataset = PlateDataset(
    IMAGE_DIR,
    LABEL_DIR,
    image_size=640,
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
)

# ============================================================
# MODEL
# ============================================================

model = PlateDetector()

model.eval()

# ============================================================
# TEST
# ============================================================

with torch.no_grad():

    batch = next(iter(loader))

    images = batch["image"]
    classes = batch["class"]
    polygons = batch["polygon"]

    output = model(images)

    print("=" * 60)
    print("INPUT")
    print("=" * 60)

    print()

    print("Images")
    print(images.shape)

    print()

    print("Classes")
    print(classes.shape)

    print(classes)

    print()

    print("Polygons")
    print(polygons.shape)

    print()

    print("=" * 60)
    print("MODEL OUTPUT")
    print("=" * 60)

    print()

    print("Class Logits")
    print(output["class_logits"].shape)

    print(output["class_logits"])

    print()

    print("Polygon Prediction")
    print(output["polygon"].shape)

    print(output["polygon"])

    print()

    print("=" * 60)
    print("SUCCESS")
    print("=" * 60)