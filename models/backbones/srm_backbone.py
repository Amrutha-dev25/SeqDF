"""
SRM residual stream backbone for predicting m2 (middle step) - partial signal
surviving as blending seams / mid-frequency artifacts in the noise residual domain.

Per your original plan: EfficientNet-B4 per-frame, then a small temporal conv
network (TCN) pools the 16 per-frame embeddings into one clip-level embedding.
EfficientNet-B4 is appropriately sized here - not too large, since the SRM signal
is narrowband and a bigger model risks overfitting to per-source noise quirks
rather than learning genuine method fingerprints (especially relevant given you
only have 93 unique source videos/identities).
"""
import torch
import torch.nn as nn


class TemporalConvPool(nn.Module):
    """Small 1D conv net over the time axis, pools T frame-embeddings -> 1 clip embedding."""
    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers += [
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, T, dim) -> (B, dim, T) for conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (B, dim)
        return x


class SRMBackbone(nn.Module):
    def __init__(self, backbone_type: str = "efficientnet_b4", embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim

        import timm
        # Note: pretrained ImageNet weights transfer reasonably well to SRM residual
        # maps since they're still locally-textured 3-channel inputs (unlike DCT maps
        # which are a fundamentally different domain - see dct_backbone.py).
        self.backbone = timm.create_model(backbone_type, pretrained=pretrained, num_classes=0)
        raw_dim = self.backbone.num_features

        self.projection = nn.Linear(raw_dim, embedding_dim) if raw_dim != embedding_dim else nn.Identity()
        self.temporal_pool = TemporalConvPool(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, T, 3, H, W) SRM residual clip
        Output: (B, embedding_dim)
        """
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        feat_flat = self.backbone(x_flat)  # (B*T, raw_dim)
        feat_flat = self.projection(feat_flat)  # (B*T, embedding_dim)
        feat = feat_flat.reshape(B, T, -1)  # (B, T, embedding_dim)
        return self.temporal_pool(feat)  # (B, embedding_dim)
