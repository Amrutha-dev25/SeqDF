"""
Full Stage A model: 3 stream backbones -> fusion MLP -> multi-label set detection
head. Predicts WHICH 3 of 5 methods were used (order-agnostic), not the order.

For order/sequence prediction within a known set, see models/stage_b_sequence/,
which is a separate, smaller model trained after this one and reusing its frozen
fused embeddings - see README and training/stageB_train_sequence.py.
"""
import torch
import torch.nn as nn

from models.backbones.rgb_backbone import RGBBackbone
from models.backbones.srm_backbone import SRMBackbone
from models.backbones.dct_backbone import DCTBackbone
from models.fusion.fusion_mlp import FusionMLP
from models.fusion.set_detection_head import SetDetectionHead


class SeqDFModel(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()

        rgb_cfg = model_cfg["rgb_backbone"]
        srm_cfg = model_cfg["srm_backbone"]
        dct_cfg = model_cfg["dct_backbone"]
        fusion_cfg = model_cfg["fusion"]
        set_head_cfg = model_cfg["set_head"]

        self.rgb_backbone = RGBBackbone(
            backbone_type=rgb_cfg["type"], embedding_dim=rgb_cfg["embedding_dim"], pretrained=rgb_cfg["pretrained"]
        )
        self.srm_backbone = SRMBackbone(
            backbone_type=srm_cfg["type"], embedding_dim=srm_cfg["embedding_dim"], pretrained=srm_cfg["pretrained"]
        )
        self.dct_backbone = DCTBackbone(
            backbone_type=dct_cfg["type"], embedding_dim=dct_cfg["embedding_dim"], pretrained=dct_cfg["pretrained"]
        )

        self.fusion = FusionMLP(
            input_dim=fusion_cfg["input_dim"], hidden_dims=fusion_cfg["hidden_dims"], dropout=fusion_cfg["dropout"]
        )

        self.set_head = SetDetectionHead(
            input_dim=self.fusion.output_dim,
            hidden_dim=set_head_cfg["hidden_dim"],
            num_methods=set_head_cfg["num_methods"],
            dropout=set_head_cfg["dropout"],
        )

    def forward(self, rgb: torch.Tensor, srm: torch.Tensor, dct: torch.Tensor, return_embedding: bool = False):
        """
        Args:
            rgb: (B, T, 3, H, W)
            srm: (B, T, 3, H, W)
            dct: (B, T, 3, H, W)
            return_embedding: if True, also returns the fused embedding -
                               needed when training Stage B on top of this model

        Returns dict with:
            set_logits: (B, 5) raw logits, one per method - use BCEWithLogitsLoss
                        directly on these during training, or sigmoid for probabilities
            fused_embedding: (B, fusion_output_dim) - only if return_embedding=True
        """
        rgb_emb = self.rgb_backbone(rgb)
        srm_emb = self.srm_backbone(srm)
        dct_emb = self.dct_backbone(dct)

        fused = self.fusion(rgb_emb, srm_emb, dct_emb)
        set_logits = self.set_head(fused)

        outputs = {
            "set_logits": set_logits,
            "rgb_embedding": rgb_emb,
            "srm_embedding": srm_emb,
            "dct_embedding": dct_emb,
        }
        if return_embedding:
            outputs["fused_embedding"] = fused
        return outputs

    def freeze_all(self):
        """Used when loading a trained Stage A model purely as a frozen feature
        extractor for Stage B training."""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
