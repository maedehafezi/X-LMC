from model.cross_attention_block import BidirectionalCrossAttention

import torch
import torch.nn as nn
import timm


class XLMC(nn.Module):
    """
    DINOv2-based model for collateral grading from biplane DSA.

    Supported inputs
    ----------------
    input_type="sequence":
        Temporal AP and LAT image sequences.

        fusion_mode="cross_attention":
            AP/LAT token interaction using bidirectional cross-attention,
            followed by temporal aggregation with a GRU.

        fusion_mode="concat":
            AP/LAT CLS tokens are concatenated directly and passed
            to the GRU. Intended as a simpler fusion baseline.

    input_type="minip":
        AP and LAT MinIP images are independently encoded with DINOv2,
        pooled, concatenated, and classified.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    task : str
        Currently only "classification" is supported.

    input_type : str
        Either "sequence" or "minip".

    fusion_mode : str
        Fusion strategy for sequence inputs:
        "cross_attention" or "concat".

    bidirectional : bool
        If True, use a bidirectional GRU.

    gru_hidden : int
        Number of hidden units in the GRU.
    """

    def __init__(
        self,
        num_classes,
        task="classification",
        input_type="sequence",
        fusion_mode="cross_attention",
        bidirectional=False,
        gru_hidden=256,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------
        self.num_classes = num_classes
        self.task = task
        self.input_type = input_type
        self.fusion_mode = fusion_mode
        self.bidirectional = bidirectional
        self.gru_hidden = gru_hidden

        if self.task != "classification":
            raise NotImplementedError(
                "Currently only task='classification' is supported."
            )

        if self.input_type not in {"sequence", "minip"}:
            raise ValueError(
                f"Unknown input_type '{self.input_type}'. "
                "Expected 'sequence' or 'minip'."
            )

        if self.fusion_mode not in {"cross_attention", "concat"}:
            raise ValueError(
                f"Unknown fusion_mode '{self.fusion_mode}'. "
                "Expected 'cross_attention' or 'concat'."
            )

        # ---------------------------------------------------------
        # Backbone
        # ---------------------------------------------------------
        self.dino_model_name = "vit_base_patch14_dinov2"

        if self.input_type == "sequence":

            self.ap_feature_extractor = self._create_dino_backbone()
            self.lat_feature_extractor = self._create_dino_backbone()

            self.dino_dim = self.ap_feature_extractor.num_features

            # -----------------------------------------------------
            # Cross-view fusion
            # -----------------------------------------------------
            if self.fusion_mode == "cross_attention":

                self.cross_attn = BidirectionalCrossAttention(
                    dim=self.dino_dim,
                    num_heads=8,
                    mlp_ratio=4.0,
                    dropout=0.1,
                )

                # AP and LAT CLS tokens are added:
                #
                # [B*T, 768] + [B*T, 768] -> [B*T, 768]
                gru_input_dim = self.dino_dim

            else:
                self.cross_attn = None

                # AP and LAT CLS tokens are concatenated:
                #
                # [B*T, 768] || [B*T, 768] -> [B*T, 1536]
                gru_input_dim = self.dino_dim * 2

            # -----------------------------------------------------
            # Temporal aggregation
            # -----------------------------------------------------
            self.gru = nn.GRU(
                input_size=gru_input_dim,
                hidden_size=gru_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            )

            gru_output_dim = gru_hidden * (2 if bidirectional else 1)

            self.classifier = self._build_classifier(
                input_dim=gru_output_dim,
                num_classes=num_classes,
            )

        # ---------------------------------------------------------
        # MinIP model
        # ---------------------------------------------------------
        else:

            self.ap_feature_extractor_minip = self._create_dino_backbone()
            self.lat_feature_extractor_minip = self._create_dino_backbone()

            self.dino_dim_minip = (
                self.ap_feature_extractor_minip.num_features
            )

            # Two DINOv2 embeddings are concatenated.
            minip_input_dim = self.dino_dim_minip * 2

            self.classifier = self._build_classifier(
                input_dim=minip_input_dim,
                num_classes=num_classes,
            )

        # Freeze pretrained DINOv2 encoders.
        self.freeze_dino()

    # =============================================================
    # Model components
    # =============================================================

    def _create_dino_backbone(self):
        """Create a pretrained DINOv2 ViT-B/14 feature extractor."""

        return timm.create_model(
            self.dino_model_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
            img_size=224,
            dynamic_img_size=True,
        )

    @staticmethod
    def _build_classifier(input_dim, num_classes):
        """Classification MLP."""

        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ELU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(128, num_classes),
        )

    # =============================================================
    # DINOv2 freezing
    # =============================================================

    def _get_dino_backbones(self):
        """Return the DINOv2 backbones used by the selected model."""

        if self.input_type == "sequence":
            return (
                self.ap_feature_extractor,
                self.lat_feature_extractor,
            )

        return (
            self.ap_feature_extractor_minip,
            self.lat_feature_extractor_minip,
        )

    def freeze_dino(self):
        """Freeze all DINOv2 backbone parameters."""

        for backbone in self._get_dino_backbones():

            for parameter in backbone.parameters():
                parameter.requires_grad = False

            backbone.eval()

    def train(self, mode=True):
        """
        Set train/eval mode while keeping frozen DINOv2 encoders
        in evaluation mode.
        """

        super().train(mode)

        for backbone in self._get_dino_backbones():
            backbone.eval()

        return self

    # =============================================================
    # Feature extraction
    # =============================================================

    @staticmethod
    def _validate_sequence_inputs(ap, lat):
        """Check AP/LAT sequence compatibility."""

        if ap.ndim != 5 or lat.ndim != 5:
            raise ValueError(
                "Sequence inputs must have shape [B, T, C, H, W]."
            )

        if ap.shape[0] != lat.shape[0]:
            raise ValueError(
                "AP and LAT sequences must have the same batch size."
            )

        if ap.shape[1] != lat.shape[1]:
            raise ValueError(
                "AP and LAT sequences must have the same number of frames."
            )

    def _extract_sequence_tokens(self, ap, lat):
        """
        Extract frozen DINOv2 tokens from AP and LAT sequences.

        Returns
        -------
        ap_feat : Tensor
            Shape [B*T, N, D]

        lat_feat : Tensor
            Shape [B*T, N, D]
        """

        B, T, C, H, W = ap.shape

        ap = ap.reshape(B * T, C, H, W)
        lat = lat.reshape(B * T, C, H, W)

        with torch.no_grad():
            ap_feat = self.ap_feature_extractor(ap)
            lat_feat = self.lat_feature_extractor(lat)

        return ap_feat, lat_feat

    # =============================================================
    # Temporal aggregation
    # =============================================================

    def _temporal_aggregation(self, features):
        """
        Aggregate frame-level features using the GRU.

        Parameters
        ----------
        features : Tensor
            Shape [B, T, D]

        Returns
        -------
        Tensor
            Final sequence representation.
        """

        _, hidden = self.gru(features)

        if self.gru.bidirectional:

            # Last forward and backward hidden states.
            feature = torch.cat(
                [hidden[-2], hidden[-1]],
                dim=-1,
            )

        else:
            feature = hidden[-1]

        return feature

    # =============================================================
    # Forward paths
    # =============================================================

    def _forward_minip(self, ap, lat):
        """Forward pass for MinIP inputs."""

        if ap.ndim != 4 or lat.ndim != 4:
            raise ValueError(
                "MinIP inputs must have shape [B, C, H, W]."
            )

        with torch.no_grad():
            ap_feat = self.ap_feature_extractor_minip(ap)
            lat_feat = self.lat_feature_extractor_minip(lat)

        # Average token representations.
        ap_feat = ap_feat.mean(dim=1)
        lat_feat = lat_feat.mean(dim=1)

        # [B, D] + [B, D] -> [B, 2D]
        features = torch.cat(
            [ap_feat, lat_feat],
            dim=-1,
        )

        return self.classifier(features)

    def _forward_sequence_cross_attention(self, ap, lat):
        """Sequence model with bidirectional cross-view attention."""

        self._validate_sequence_inputs(ap, lat)

        B, T = ap.shape[:2]

        ap_tokens, lat_tokens = self._extract_sequence_tokens(
            ap,
            lat,
        )

        (
            ap_tokens_updated,
            lat_tokens_updated,
            _,
            _,
        ) = self.cross_attn(
            ap_tokens,
            lat_tokens,
        )

        # Updated CLS tokens.
        ap_cls = ap_tokens_updated[:, 0, :]
        lat_cls = lat_tokens_updated[:, 0, :]

        # Cross-view fusion.
        # Shape: [B*T, D]
        fused = ap_cls + lat_cls

        # Restore temporal dimension.
        # Shape: [B, T, D]
        fused = fused.reshape(B, T, self.dino_dim)

        temporal_feature = self._temporal_aggregation(fused)

        return self.classifier(temporal_feature)

    def _forward_sequence_concat(self, ap, lat):
        """Sequence model with direct AP/LAT feature concatenation."""

        self._validate_sequence_inputs(ap, lat)

        B, T = ap.shape[:2]

        ap_tokens, lat_tokens = self._extract_sequence_tokens(
            ap,
            lat,
        )

        # DINOv2 CLS tokens.
        ap_cls = ap_tokens[:, 0, :]
        lat_cls = lat_tokens[:, 0, :]

        ap_cls = ap_cls.reshape(B, T, self.dino_dim)
        lat_cls = lat_cls.reshape(B, T, self.dino_dim)

        # Shape: [B, T, 2D]
        fused = torch.cat(
            [ap_cls, lat_cls],
            dim=-1,
        )

        temporal_feature = self._temporal_aggregation(fused)

        return self.classifier(temporal_feature)

    # =============================================================
    # Forward
    # =============================================================

    def forward(self, x, x2):
        """
        Parameters
        ----------
        x :
            AP input.

        x2 :
            LAT input.
        """

        if self.input_type == "minip":
            return self._forward_minip(x, x2)

        if self.fusion_mode == "cross_attention":
            return self._forward_sequence_cross_attention(x, x2)

        return self._forward_sequence_concat(x, x2)