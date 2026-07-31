import torch

# ===== sửa tên file train của bạn =====
from train import RSNAClassifier, Config


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def print_model_info(head_type):
    config = Config()

    model = RSNAClassifier(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        head_type=head_type,
        mlp_hidden_dim=config.MLP_HIDDEN_DIM,
        dropout=config.DROPOUT,

        kan_hidden_dim=config.KAN_HIDDEN_DIM,
        kan_grid_size=config.KAN_GRID_SIZE,
        kan_grid_min=config.KAN_GRID_MIN,
        kan_grid_max=config.KAN_GRID_MAX,

        kan_official_hidden_dim=config.KAN_OFFICIAL_HIDDEN_DIM,
        kan_official_grid=config.KAN_OFFICIAL_GRID,
        kan_official_k=config.KAN_OFFICIAL_K,
        kan_official_seed=config.KAN_OFFICIAL_SEED,
        kan_official_speed_mode=config.KAN_OFFICIAL_SPEED_MODE,
    )

    print("=" * 70)
    print(f"{head_type}")

    total = count_params(model)
    backbone = count_params(model.backbone)

    print(f"Backbone Parameters : {backbone:,}")

    # in toàn bộ module con
    print("\nSubmodules:")
    for name, module in model.named_children():
        print(f"  {name:20s}: {count_params(module):,}")

    print(f"\nTotal Parameters    : {total:,}")
    print(f"Head Parameters     : {total - backbone:,}")
    print()


for head in [
    "linear",
    "mlp",
    "kan_custom",
    "kan_official",
]:
    print_model_info(head)