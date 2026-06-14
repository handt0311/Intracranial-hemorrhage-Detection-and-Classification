from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


class ResNet18FPNBackbone(nn.Module):
    """
    ResNet18 convolutional backbone with an FPN for Faster R-CNN.

    The ResNet18 convolutional layers are initialized from the RSNA classification
    model. The classification head is not used in Stage 2.
    """

    def __init__(self, resnet18_body: nn.Module, out_channels: int = 256):
        super().__init__()

        self.body = IntermediateLayerGetter(
            resnet18_body,
            return_layers={
                "layer1": "0",
                "layer2": "1",
                "layer3": "2",
                "layer4": "3",
            },
        )

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256, 512],
            out_channels=out_channels,
        )

        self.out_channels = out_channels

    def forward(self, x):
        features = self.body(x)
        features = self.fpn(features)
        return features


def build_resnet18_1ch():
    """
    Build a ResNet18 feature extractor with a 1-channel input layer.
    This matches the RSNA classification backbone.
    """
    resnet = torchvision.models.resnet18(weights=None)

    resnet.conv1 = nn.Conv2d(
        1,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

    resnet.fc = nn.Identity()

    return resnet


def load_rsna_backbone_weights(resnet18_body: nn.Module, checkpoint_path: str):
    """
    Load only the ResNet18 convolutional backbone from the RSNA classifier.

    Expected checkpoint keys:
        backbone.conv1.weight
        backbone.bn1.*
        backbone.layer1.*
        ...
        classifier.*
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"[WARN] RSNA checkpoint not found: {checkpoint_path}")
        print("[WARN] Using randomly initialized ResNet18 backbone.")
        return resnet18_body

    print(f"Loading RSNA classification checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    backbone_state = {}

    for key, value in state_dict.items():
        clean_key = key

        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]

        if not clean_key.startswith("backbone."):
            continue

        clean_key = clean_key[len("backbone."):]

        if clean_key.startswith("fc."):
            continue

        backbone_state[clean_key] = value

    if len(backbone_state) == 0:
        print("[WARN] No backbone.* keys found in checkpoint.")
        print("[WARN] Using randomly initialized ResNet18 backbone.")
        return resnet18_body

    missing, unexpected = resnet18_body.load_state_dict(backbone_state, strict=False)

    print(f"Loaded backbone tensors: {len(backbone_state)}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("Missing examples:", missing[:10])

    if len(unexpected) > 0:
        print("Unexpected examples:", unexpected[:10])

    return resnet18_body


def build_resnet18_fasterrcnn(
    num_classes: int,
    rsna_checkpoint_path: str = None,
    image_size: int = 512,
):
    """
    Build Faster R-CNN with a ResNet18-FPN backbone.

    Class IDs:
        0 = background
        1 = epidural
        2 = intraparenchymal
        3 = intraventricular
        4 = subarachnoid
        5 = subdural
    """
    resnet18_body = build_resnet18_1ch()

    if rsna_checkpoint_path is not None and str(rsna_checkpoint_path) != "":
        resnet18_body = load_rsna_backbone_weights(
            resnet18_body=resnet18_body,
            checkpoint_path=rsna_checkpoint_path,
        )

    backbone = ResNet18FPNBackbone(
        resnet18_body=resnet18_body,
        out_channels=256,
    )

    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        image_mean=[0.0],
        image_std=[1.0],
        min_size=image_size,
        max_size=image_size,
    )

    return model