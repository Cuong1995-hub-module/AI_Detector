"""
model.py - PlateDetector V1
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + identity
        return self.relu(x)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ResidualBlock(64),
            ConvBlock(64, 128),
            ResidualBlock(128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.net(x)
        return torch.flatten(x, 1)


class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class PolygonHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 8),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class PlateDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FeatureExtractor()
        self.cls_head = ClassificationHead()
        self.poly_head = PolygonHead()

    def forward(self, images):
        features = self.backbone(images)
        return {
            "class_logits": self.cls_head(features),
            "polygon": self.poly_head(features),
        }


if __name__ == "__main__":
    model = PlateDetector()
    dummy = torch.randn(4, 3, 640, 640)
    output = model(dummy)
    print(output["class_logits"].shape)
    print(output["polygon"].shape)
