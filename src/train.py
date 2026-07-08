"""
train.py
Training script for PlateDetector V1
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset import PlateDataset
from model import PlateDetector
from loss import PlateLoss

# ============================================================
# CONFIG
# ============================================================

TRAIN_IMAGE_DIR = "/content/Dataset_Motorcycle/images/train"
TRAIN_LABEL_DIR = "/content/Dataset_Motorcycle/labels/train"

VAL_IMAGE_DIR = "/content/Dataset_Motorcycle/images/val"
VAL_LABEL_DIR = "/content/Dataset_Motorcycle/labels/val"

IMAGE_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# DATASET
# ============================================================

train_dataset = PlateDataset(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, IMAGE_SIZE)
val_dataset = PlateDataset(VAL_IMAGE_DIR, VAL_LABEL_DIR, IMAGE_SIZE)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# ============================================================
# MODEL
# ============================================================

model = PlateDetector().to(DEVICE)
criterion = PlateLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")

print(f"\nDevice : {DEVICE}")

# ============================================================
# TRAIN LOOP
# ============================================================

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    for batch in train_loader:

        images = batch["image"].to(DEVICE)
        classes = batch["class"].to(DEVICE)
        polygons = batch["polygon"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        losses = criterion(
            outputs,
            {
                "class": classes,
                "polygon": polygons,
            },
        )

        loss = losses["total_loss"]

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ================= Validation =================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(DEVICE)
            classes = batch["class"].to(DEVICE)
            polygons = batch["polygon"].to(DEVICE)

            outputs = model(images)

            losses = criterion(
                outputs,
                {
                    "class": classes,
                    "polygon": polygons,
                },
            )

            val_loss += losses["total_loss"].item()

    val_loss /= len(val_loader)

    print(
        f"Epoch [{epoch+1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    torch.save(model.state_dict(), "last_model.pth")

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(model.state_dict(), "best_model.pth")

        print("  -> Best model saved.")

print("\nTraining Finished.")
