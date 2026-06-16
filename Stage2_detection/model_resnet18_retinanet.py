from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.rpn import AnchorGenerator

from Stage2_detection.model_resnet18_fasterrcnn import (
    ResNet18FPNBackbone,
    build_resnet18_1ch,
    load_rsna_backbone_weights,
)


def build_resnet18_retinanet(
    num_classes: int,
    rsna_checkpoint_path: str = None,
    image_size: int = 512,
):
    """
    Build RetinaNet with a ResNet18-FPN backbone initialized from
    the RSNA KAN-based classification checkpoint.

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

    model = RetinaNet(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        image_mean=[0.0],
        image_std=[1.0],
        min_size=image_size,
        max_size=image_size,
    )

    return model
