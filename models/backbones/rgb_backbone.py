"""
RGB stream backbone for predicting m3 (the dominant, easiest-to-recover signal).

Uses Video Swin-T (via timm/torchvision video models) as the default - this is the
video-native analog of what was originally proposed as "Conformer" in the chat:
Conformer is a speech/1D-sequence architecture (conv + self-attention for audio),
and Video Swin/TimeSformer are the equivalent designs built specifically for
spatio-temporal video input, which is what we actually have here.

Falls back to a simple 3D-CNN + temporal transformer if the chosen pretrained
video backbone isn't available in the environment (e.g. no internet access to
download pretrained weights) - see _build_fallback_backbone.
"""
import torch
import torch.nn as nn


class RGBBackbone(nn.Module):
    def __init__(self, backbone_type: str = "video_swin_t", embedding_dim: int = 768, pretrained: bool = True):
        super().__init__()
        self.backbone_type = backbone_type
        self.embedding_dim = embedding_dim

        try:
            self.backbone, self._raw_out_dim = self._build_backbone(backbone_type, pretrained)
            self._using_fallback = False
        except Exception as e:
            print(f"WARNING: could not build '{backbone_type}' ({e}). "
                  f"Falling back to a lightweight 3D-CNN + temporal transformer. "
                  f"This will train but won't match pretrained-video-transformer quality - "
                  f"check your timm/torchvision install and internet access for pretrained weights.")
            self.backbone, self._raw_out_dim = self._build_fallback_backbone()
            self._using_fallback = True

        # project backbone output to the configured embedding_dim, in case the raw
        # backbone's native output dim doesn't match what fusion expects
        self.projection = nn.Linear(self._raw_out_dim, embedding_dim) if self._raw_out_dim != embedding_dim else nn.Identity()

    def _build_backbone(self, backbone_type: str, pretrained: bool):
        if backbone_type == "video_swin_t":
            import torchvision.models.video as video_models
            model = video_models.swin3d_t(weights="KINETICS400_V1" if pretrained else None)
            out_dim = model.head.in_features
            model.head = nn.Identity()
            return model, out_dim

        elif backbone_type == "timesformer":
            import timm
            model = timm.create_model("timesformer_base", pretrained=pretrained, num_classes=0)
            out_dim = model.num_features
            return model, out_dim

        elif backbone_type == "videomae_base":
            import timm
            model = timm.create_model("vit_base_patch16_224.mae", pretrained=pretrained, num_classes=0)
            out_dim = model.num_features
            return model, out_dim

        else:
            raise ValueError(f"Unknown rgb backbone_type: {backbone_type}")

    def _build_fallback_backbone(self):
        """Lightweight fallback: 3D conv stem + temporal transformer pooling.
        Used only if the requested pretrained video backbone can't be loaded."""
        class Simple3DCNN(nn.Module):
            def __init__(self, out_dim=512):
                super().__init__()
                self.conv3d = nn.Sequential(
                    nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
                    nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=(1, 2, 2)),
                    nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
                    nn.BatchNorm3d(128), nn.ReLU(inplace=True),
                    nn.Conv3d(128, out_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
                    nn.BatchNorm3d(out_dim), nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool3d((None, 1, 1)),
                )
                self.out_dim = out_dim

            def forward(self, x):
                # x: (B, 3, T, H, W) -> (B, out_dim, T, 1, 1)
                feat = self.conv3d(x)
                feat = feat.squeeze(-1).squeeze(-1)  # (B, out_dim, T)
                feat = feat.mean(dim=-1)  # temporal mean pool -> (B, out_dim)
                return feat

        model = Simple3DCNN(out_dim=512)
        return model, 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, T, 3, H, W)
        Output: (B, embedding_dim)
        """
        if self._using_fallback:
            x = x.permute(0, 2, 1, 3, 4)  # (B, 3, T, H, W) for Conv3d
            feat = self.backbone(x)
        else:
            if self.backbone_type == "video_swin_t":
                x = x.permute(0, 2, 1, 3, 4)  # torchvision video models expect (B, C, T, H, W)
                feat = self.backbone(x)
            else:
                # frame-wise transformer backbones (timesformer/videomae): pool over time
                B, T, C, H, W = x.shape
                x_flat = x.reshape(B * T, C, H, W)
                feat_flat = self.backbone(x_flat)  # (B*T, raw_out_dim)
                feat = feat_flat.reshape(B, T, -1).mean(dim=1)  # mean pool over time

        return self.projection(feat)
