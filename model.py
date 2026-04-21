import math
import torch
import torch.nn as nn
import torchvision.models as models


class LinearHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class SimpleKANHead(nn.Module):
    """
    A practical KAN-style head:
    - expands features with a Gaussian basis
    - then passes them through a small MLP to produce logits
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_basis: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_basis = num_basis

        centers = torch.linspace(-2.0, 2.0, num_basis)
        self.register_buffer("centers", centers)

        self.log_sigma = nn.Parameter(
            torch.tensor(math.log(0.5), dtype=torch.float32)
        )

        expanded_dim = in_features * num_basis

        self.net = nn.Sequential(
            nn.Linear(expanded_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, D]
        sigma = torch.exp(self.log_sigma).clamp(min=1e-3)

        # stabilize basis expansion
        x_norm = torch.tanh(x)

        # [B, D, 1] - [K] -> [B, D, K]
        basis = torch.exp(
            -((x_norm.unsqueeze(-1) - self.centers) ** 2) / (2 * sigma ** 2)
        )

        # [B, D*K]
        basis = basis.flatten(start_dim=1)

        return self.net(basis)


class RSNAClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = False,
        head_type: str = "mlp",
        mlp_hidden_dim: int = 256,
        dropout: float = 0.2,
        kan_num_basis: int = 8,
        kan_hidden_dim: int = 256,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        old_conv1 = backbone.conv1

        # Replace 3-channel conv with 1-channel conv
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            # Convert pretrained RGB conv weights -> grayscale by averaging channels
            with torch.no_grad():
                backbone.conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(
                backbone.conv1.weight, mode="fan_out", nonlinearity="relu"
            )

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone

        head_type = head_type.lower()
        self.head_type = head_type

        if head_type == "linear":
            self.classifier = LinearHead(in_features, num_classes)
        elif head_type == "mlp":
            self.classifier = MLPHead(
                in_features=in_features,
                num_classes=num_classes,
                hidden_dim=mlp_hidden_dim,
                dropout=dropout,
            )
        elif head_type == "kan":
            self.classifier = SimpleKANHead(
                in_features=in_features,
                num_classes=num_classes,
                num_basis=kan_num_basis,
                hidden_dim=kan_hidden_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return logits