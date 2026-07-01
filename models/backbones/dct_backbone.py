"""
DCT energy map stream backbone for predicting m1 (first/earliest step) - the
weakest signal in the whole pipeline, surviving only as deep frequency-domain
fingerprints after two further rounds of blending on top.

Deliberately small (ResNet-18, modified first conv) and NOT ImageNet-pretrained
by default: DCT energy maps are not natural images (no edges/textures/objects in
the visual sense ImageNet teaches), so ImageNet pretrained features transfer
poorly here and can actively bias the model toward irrelevant priors. Better to
train this one from scratch on your own data, even though that means it needs
more epochs/data to converge - given how weak m1's signal already is, don't
compound that with a mismatched pretrained prior.

A 2-layer transformer (not just average pooling) is used for temporal pooling
here, on the theory that *which* frames show frequency anomalies (not just
their average) may itself be informative for this hardest of the three targets.
"""
import torch
import torch.nn as nn


class TemporalTransformerPool(nn.Module):
    """Pools T per-frame embeddings into one clip embedding via self-attention +
    a learned [CLS]-style pooling token, rather than simple averaging."""
    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B, T, dim)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, dim)
        x = self.transformer(x)
        return x[:, 0, :]  # take pooled CLS token -> (B, dim)


class DCTBackbone(nn.Module):
    def __init__(self, backbone_type: str = "resnet18_modified", embedding_dim: int = 512, pretrained: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim

        import torchvision.models as tv_models
        # Build plain resnet18; pretrained=False by default per the reasoning above.
        # If pretrained=True is explicitly requested anyway, we still load it but the
        # first conv layer is generic enough (3-channel input) that no surgery is
        # needed - DCT maps are stacked as 3 pseudo-RGB channels already.
        self.backbone = tv_models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        raw_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.projection = nn.Linear(raw_dim, embedding_dim) if raw_dim != embedding_dim else nn.Identity()
        self.temporal_pool = TemporalTransformerPool(embedding_dim, num_layers=2, num_heads=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, T, 3, H, W) DCT energy map clip
        Output: (B, embedding_dim)
        """
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        feat_flat = self.backbone(x_flat)  # (B*T, raw_dim)
        feat_flat = self.projection(feat_flat)
        feat = feat_flat.reshape(B, T, -1)  # (B, T, embedding_dim)
        return self.temporal_pool(feat)  # (B, embedding_dim)
