from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class PlateDataset(Dataset):

    def __init__(self, image_dir, label_dir, image_size=640):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size

        self.images = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            self.images.extend(self.image_dir.glob(ext))
        self.images = sorted(self.images)

        print("=" * 60)
        print("PlateDataset V3")
        print("=" * 60)
        print("Image Folder :", self.image_dir)
        print("Label Folder :", self.label_dir)
        print("Image Size   :", self.image_size)
        print("Total Images :", len(self.images))
        print("=" * 60)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label_path = self.label_dir / (img_path.stem + ".txt")

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype("float32") / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        if not label_path.exists():
            raise RuntimeError(f"Missing label: {label_path}")

        with open(label_path) as f:
            data = f.readline().strip().split()

        if len(data) != 9:
            raise RuntimeError(f"Invalid label format: {label_path}")

        cls = int(data[0])
        polygon = list(map(float, data[1:]))

        if len(polygon) != 8:
            raise RuntimeError(f"Invalid polygon: {label_path}")

        return {
            "image": image,
            "class": torch.tensor(cls, dtype=torch.long),
            "polygon": torch.tensor(polygon, dtype=torch.float32),
            "image_path": str(img_path),
            "original_width": w,
            "original_height": h,
        }


if __name__ == "__main__":
    dataset = PlateDataset(
        "/media/cuong/PROJECTS/Dataset_Single/images/train",
        "/media/cuong/PROJECTS/Dataset_Single/labels/train",
    )

    sample = dataset[0]

    print("\nDataset Size:", len(dataset))
    print(sample.keys())
    print("Image Shape:", sample["image"].shape)
    print("Class:", sample["class"])
    print("Polygon:", sample["polygon"])
    print("Image:", sample["image_path"])
