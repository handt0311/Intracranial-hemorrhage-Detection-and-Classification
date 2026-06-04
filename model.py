import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Optional import:
# - linear / mlp / custom kan can still run without pykan installed
# - official KAN head requires: pip install pykan
try:
    from kan import KAN as PyKAN
except ImportError as e:
    PyKAN = None
    _PYKAN_IMPORT_ERROR = e


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
        dropout: float = 0.2,
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


class KANLayer(nn.Module):
    """
    Custom dependency-free KAN-inspired layer.

    This is the old/custom KAN-like implementation:
    - each edge has a learnable 1D function
    - node outputs are sums of edge activations
    - spline is implemented as a piecewise-linear spline on a fixed grid

    Note:
        This is NOT a full official pykan reproduction.
        Keep this for comparison as head_type="kan" or "kan_custom".
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int = 16,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.grid_min = grid_min
        self.grid_max = grid_max

        # Uniform grid shared structurally, but each edge has its own spline values.
        grid = torch.linspace(grid_min, grid_max, grid_size)
        self.register_buffer("grid", grid)

        # Learnable spline values for each edge at each grid knot.
        # Shape: [out_dim, in_dim, grid_size]
        self.spline_values = nn.Parameter(
            0.01 * torch.randn(out_dim, in_dim, grid_size)
        )

        # Residual/base branch coefficients for each edge.
        self.base_scale = nn.Parameter(torch.ones(out_dim, in_dim))
        self.spline_scale = nn.Parameter(torch.ones(out_dim, in_dim))

    def _piecewise_linear_spline(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate edge-wise learnable piecewise-linear spline.

        Args:
            x: Tensor of shape [B, in_dim]

        Returns:
            spline_out: Tensor of shape [B, out_dim, in_dim]
        """
        bsz = x.size(0)

        # Clamp inputs to the spline support range.
        x = x.clamp(self.grid_min, self.grid_max)

        # Normalize x into [0, grid_size - 1].
        t = (x - self.grid_min) / (self.grid_max - self.grid_min) * (
            self.grid_size - 1
        )

        # Left/right knot indices for linear interpolation.
        left_idx = torch.floor(t).long().clamp(0, self.grid_size - 2)
        right_idx = left_idx + 1

        # Interpolation weight.
        alpha = (t - left_idx.float()).unsqueeze(1)  # [B, 1, in_dim]

        # Expand edge spline table to batch.
        # spline_values: [out_dim, in_dim, grid_size]
        spline_table = self.spline_values.unsqueeze(0).expand(bsz, -1, -1, -1)
        # [B, out_dim, in_dim, grid_size]

        left_idx_exp = left_idx.unsqueeze(1).unsqueeze(-1).expand(
            -1, self.out_dim, -1, 1
        )
        right_idx_exp = right_idx.unsqueeze(1).unsqueeze(-1).expand(
            -1, self.out_dim, -1, 1
        )

        left_val = torch.gather(spline_table, dim=3, index=left_idx_exp).squeeze(-1)
        right_val = torch.gather(spline_table, dim=3, index=right_idx_exp).squeeze(-1)

        spline_out = (1.0 - alpha) * left_val + alpha * right_val
        return spline_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_dim]

        Returns:
            y: [B, out_dim]
        """
        # Base branch on each edge: SiLU applied to each input feature.
        base = F.silu(x).unsqueeze(1)  # [B, 1, in_dim]

        # Learnable spline branch on each edge.
        spline = self._piecewise_linear_spline(x)  # [B, out_dim, in_dim]

        # Combine base branch and spline branch per edge.
        edge_out = (
            self.base_scale.unsqueeze(0) * base
            + self.spline_scale.unsqueeze(0) * spline
        )  # [B, out_dim, in_dim]

        # KAN node operation: sum incoming edge activations.
        y = edge_out.sum(dim=2)  # [B, out_dim]
        return y


class TrueKANHead(nn.Module):
    """
    Old/custom KAN-inspired head for RSNA classification.

    Design:
    - input: feature vector from ResNet18 backbone
    - custom KAN-like layer 1: in_features -> hidden_dim
    - custom KAN-like layer 2: hidden_dim -> num_classes

    Use this with:
        head_type="kan"
        head_type="kan_custom"
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        grid_size: int = 16,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.layer1 = KANLayer(
            in_dim=in_features,
            out_dim=hidden_dim,
            grid_size=grid_size,
            grid_min=grid_min,
            grid_max=grid_max,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer2 = KANLayer(
            in_dim=hidden_dim,
            out_dim=num_classes,
            grid_size=grid_size,
            grid_min=grid_min,
            grid_max=grid_max,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize backbone features to keep spline inputs in a stable range.
        x = torch.tanh(x)

        x = self.layer1(x)
        x = self.norm1(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.layer2(x)
        return logits


class OfficialKANHead(nn.Module):
    """
    Official pykan-based KAN head.

    This is the KAN head you should use when you want a standard KAN head.

    Design:
        input feature vector from ResNet18 backbone: [B, in_features]
        official pykan KAN:
            KAN(width=[in_features, hidden_dim, num_classes], grid=grid, k=k)
        output logits: [B, num_classes]

    Use this with:
        head_type="kan_official"
        head_type="official_kan"
        head_type="pykan"

    Requirement:
        pip install pykan
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 16,
        grid: int = 5,
        k: int = 3,
        seed: int = 42,
        speed_mode: bool = True,
    ):
        super().__init__()

        if PyKAN is None:
            raise ImportError(
                "OfficialKANHead requires pykan. Install it with: pip install pykan"
            ) from _PYKAN_IMPORT_ERROR

        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.grid = grid
        self.k = k
        self.seed = seed

        self.kan = PyKAN(
            width=[in_features, hidden_dim, num_classes],
            grid=grid,
            k=k,
            seed=seed,
        )

        # Disable symbolic branch for faster normal PyTorch training.
        # This is recommended when you only need numerical training.
        if speed_mode and hasattr(self.kan, "speed"):
            speed_result = self.kan.speed()
            if speed_result is not None:
                self.kan = speed_result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.kan(x)
        return logits

    @torch.no_grad()
    def update_grid_from_samples(self, x: torch.Tensor):
        """
        Optional helper.

        You can call this before training or during training if you want pykan
        to adapt its spline grids to real CNN features.

        Example:
            features = model.backbone(images)
            model.classifier.update_grid_from_samples(features)
        """
        if hasattr(self.kan, "update_grid_from_samples"):
            return self.kan.update_grid_from_samples(x)

        raise AttributeError(
            "The installed pykan version does not have update_grid_from_samples()."
        )


class RSNAClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = False,
        head_type: str = "mlp",
        mlp_hidden_dim: int = 256,
        dropout: float = 0.2,

        # Old/custom KAN head parameters.
        # Used by head_type="kan" or "kan_custom".
        kan_hidden_dim: int = 64,
        kan_grid_size: int = 16,
        kan_grid_min: float = -2.0,
        kan_grid_max: float = 2.0,

        # Official pykan KAN head parameters.
        # Used by head_type="kan_official", "official_kan", or "pykan".
        kan_official_hidden_dim: int = 16,
        kan_official_grid: int = 5,
        kan_official_k: int = 3,
        kan_official_seed: int = 42,
        kan_official_speed_mode: bool = True,

        # Absorb unused legacy arguments safely, e.g. kan_num_basis from older files.
        **unused_kwargs,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        old_conv1 = backbone.conv1

        # Replace the original 3-channel conv with a 1-channel conv for CT slices.
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            # Convert pretrained RGB weights to grayscale by averaging channels.
            with torch.no_grad():
                backbone.conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(
                backbone.conv1.weight,
                mode="fan_out",
                nonlinearity="relu",
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

        elif head_type in ["kan", "kan_custom", "custom_kan", "true_kan"]:
            self.classifier = TrueKANHead(
                in_features=in_features,
                num_classes=num_classes,
                hidden_dim=kan_hidden_dim,
                grid_size=kan_grid_size,
                grid_min=kan_grid_min,
                grid_max=kan_grid_max,
                dropout=dropout,
            )

        elif head_type in ["kan_official", "official_kan", "pykan"]:
            self.classifier = OfficialKANHead(
                in_features=in_features,
                num_classes=num_classes,
                hidden_dim=kan_official_hidden_dim,
                grid=kan_official_grid,
                k=kan_official_k,
                seed=kan_official_seed,
                speed_mode=kan_official_speed_mode,
            )

        else:
            raise ValueError(
                f"Unsupported head_type: {head_type}. "
                f"Choose from: 'linear', 'mlp', 'kan', 'kan_custom', "
                f"'kan_official', 'official_kan', 'pykan'."
            )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return logits