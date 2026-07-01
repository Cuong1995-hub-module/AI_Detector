import torch
from torch.utils.data import DataLoader

from dataset import PlateDataset

# ============================================================
# DATASET PATH
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

print()
print("Dataset Size :", len(dataset))
print()

# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)

# ============================================================
# TEST
# ============================================================

for i, batch in enumerate(loader):

    print("=" * 60)
    print(f"Batch {i + 1}")
    print("=" * 60)

    print()

    print("Images")
    print(batch["image"].shape)

    print()

    print("Classes")
    print(batch["class"].shape)
    print(batch["class"])

    print()

    print("Polygons")
    print(batch["polygon"].shape)

    print()

    print("First Polygon")
    print(batch["polygon"][0])

    print()

    print("Image Path")
    print(batch["image_path"][0])

    break