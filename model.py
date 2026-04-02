import torch
import torch.nn as nn
import torchvision.models as models


class RSNAClassifier(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()

        self.backbone = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)

        # Change first conv to accept 1-channel image
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat).squeeze(1)
        return logits