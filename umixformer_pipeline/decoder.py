"""
U-MixFormer Decoder — inspired by Yeom & von Klitzing (2023).

Key ideas implemented:
  1. Mix-Attention: Keys/Values come from a MIX of encoder + previous decoder
     stage features, not just one source.
  2. Queries come from lateral encoder connections (not skip-connections).
  3. U-Net-like progressive refinement: decoder starts from the deepest
     (most contextual) features and progressively incorporates finer details.
  4. All decoder stage outputs are concatenated for the final prediction MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================================
# Mix-Attention Module
# ============================================================================

class MixAttention(nn.Module):
    """Mix-Attention from U-MixFormer.

    Queries come from the lateral encoder feature (Xq).
    Keys and Values come from a MIXED set of features (Xkv) —
    a concatenation of spatially-aligned encoder + decoder features.

    This allows the query to find matches across ALL stages
    (different contextual granularities), enabling richer feature refinement.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_q  : (B, N_q, C)  — query tokens from lateral encoder feature
        x_kv : (B, N_kv, C) — mixed key/value tokens from multiple stages

        Returns
        -------
        out : (B, N_q, C)
        """
        B, Nq, C = x_q.shape

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


# ============================================================================
# FFN (Feed-Forward Network with DWConv — from SegFormer / U-MixFormer)
# ============================================================================

class MixFFN(nn.Module):
    """Mix-FFN: Linear → DWConv3×3 → GELU → Linear.

    The depthwise conv injects positional/local information without
    explicit positional encoding.
    """

    def __init__(self, dim: int, expansion: int = 4, drop: float = 0.0):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """x: (B, H*W, C) → (B, H*W, C)"""
        B, N, C = x.shape
        x = self.fc1(x)
        # Reshape to spatial for DWConv
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.dwconv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================================
# Decoder Stage (Transformer Block with Mix-Attention)
# ============================================================================

class DecoderStage(nn.Module):
    """Single U-MixFormer decoder stage.

    Structure (following Eq. 5 in the paper):
        A_i = LN(MixAtt(LN(Xkv), LN(Xq))) + LN(Xq)   # residual
        D_i = FFN(A_i) + A_i                             # residual

    Self-attention is discarded (as in FeedFormer / U-MixFormer).
    """

    def __init__(self, dim: int, num_heads: int = 8, ffn_expansion: int = 4,
                 attn_drop: float = 0.0, ffn_drop: float = 0.1,
                 depth: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "norm_q": nn.LayerNorm(dim),
                "norm_kv": nn.LayerNorm(dim),
                "mix_attn": MixAttention(dim, num_heads, attn_drop, attn_drop),
                "norm_ff": nn.LayerNorm(dim),
                "ffn": MixFFN(dim, ffn_expansion, ffn_drop),
            }))

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor,
                H: int, W: int) -> torch.Tensor:
        """
        x_q  : (B, H*W, C) — from lateral encoder
        x_kv : (B, N_kv, C) — mixed features
        Returns: (B, H*W, C)
        """
        x = x_q
        for blk in self.blocks:
            # Mix-attention with residual
            q_normed = blk["norm_q"](x)
            kv_normed = blk["norm_kv"](x_kv)
            x = blk["mix_attn"](q_normed, kv_normed) + x

            # FFN with residual
            x = blk["ffn"](blk["norm_ff"](x), H, W) + x

        return x


# ============================================================================
# Feature Alignment Module
# ============================================================================

class FeatureAligner(nn.Module):
    """Align multi-scale features to a target spatial size and channel dim.

    Uses adaptive average pooling + linear projection (Eq. 3 in paper).
    """

    def __init__(self, in_channels_list: list[int], out_channels: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, out_channels),
            )
            for c in in_channels_list
        ])

    def forward(self, features: list[torch.Tensor], target_h: int,
                target_w: int) -> torch.Tensor:
        """Align and concatenate features along sequence dimension.

        Parameters
        ----------
        features : list of (B, C_i, H_i, W_i) tensors
        target_h, target_w : target spatial dimensions

        Returns
        -------
        mixed : (B, N_total, out_channels)  where N_total = sum(H_i'*W_i')
        """
        aligned = []
        for feat, proj in zip(features, self.projections):
            B, C, H, W = feat.shape
            # Spatially align to target size
            if H != target_h or W != target_w:
                feat = F.adaptive_avg_pool2d(feat, (target_h, target_w))
            # Flatten spatial → sequence
            feat = rearrange(feat, "b c h w -> b (h w) c")
            feat = proj(feat)
            aligned.append(feat)
        # Concatenate along sequence dimension
        return torch.cat(aligned, dim=1)


# ============================================================================
# U-MixFormer Decoder
# ============================================================================

class UMixFormerDecoder(nn.Module):
    """Full U-MixFormer decoder.

    N decoder stages, each using mix-attention where:
      - Queries = lateral encoder features
      - Keys/Values = mix of encoder + previous decoder outputs

    Feature set selection (Eq. 2):
      Stage 1: F_1 = {E_1, E_2, E_3, E_4}     (all encoder)
      Stage 2: F_2 = {E_1, E_2, E_3, D_4}      (replace deepest with decoder out)
      Stage 3: F_3 = {E_1, E_2, D_3, D_4}
      Stage 4: F_4 = {E_1, D_2, D_3, D_4}

    All decoder outputs are upsampled and concatenated for final prediction.
    """

    def __init__(
        self,
        encoder_channels: list[int],
        decoder_dim: int = 256,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        attn_drop: float = 0.0,
        ffn_drop: float = 0.1,
        decoder_depth: int = 1,
        num_classes: int = 4,
    ):
        super().__init__()
        self.N = len(encoder_channels)  # number of stages
        self.decoder_dim = decoder_dim

        # Project each encoder feature to unified decoder dim
        self.encoder_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, decoder_dim, 1),
                nn.BatchNorm2d(decoder_dim),
                nn.GELU(),
            )
            for c in encoder_channels
        ])

        # Feature aligners for each decoder stage
        # Stage i mixes N features, all projected to decoder_dim
        self.feature_aligners = nn.ModuleList([
            FeatureAligner([decoder_dim] * self.N, decoder_dim)
            for _ in range(self.N)
        ])

        # Decoder stages (process from deepest → shallowest)
        self.decoder_stages = nn.ModuleList([
            DecoderStage(
                dim=decoder_dim,
                num_heads=num_heads,
                ffn_expansion=ffn_expansion,
                attn_drop=attn_drop,
                ffn_drop=ffn_drop,
                depth=decoder_depth,
            )
            for _ in range(self.N)
        ])

        # Final prediction MLP: concat all decoder outputs → class logits
        self.final_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * self.N, decoder_dim, 1),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        encoder_features : list of (B, C_i, H_i, W_i), i=0..N-1
            Multi-scale encoder outputs (index 0 = shallowest/largest).

        Returns
        -------
        logits : (B, num_classes, H/4, W/4)
        """
        B = encoder_features[0].shape[0]
        N = self.N

        # Project encoder features to decoder_dim
        enc_proj = [proj(feat) for proj, feat in
                    zip(self.encoder_projections, encoder_features)]

        # Decoder outputs storage (initially None)
        dec_outputs = [None] * N

        # Process from deepest stage to shallowest (stage index i=0..N-1)
        # i=0 processes the deepest encoder feature (index N-1)
        for i in range(N):
            # Encoder stage index for query (reversed: deepest first)
            enc_idx = N - 1 - i

            # Query = lateral encoder feature
            x_q = enc_proj[enc_idx]
            Hq, Wq = x_q.shape[2], x_q.shape[3]
            x_q_flat = rearrange(x_q, "b c h w -> b (h w) c")

            # Build feature set F_i (Eq. 2 from paper)
            # For stage i: use encoder features for stages not yet decoded,
            # use decoder outputs for stages already decoded
            feature_set = []
            for j in range(N):
                if j > enc_idx and dec_outputs[j] is not None:
                    # Use decoder output for already-decoded stages
                    feature_set.append(dec_outputs[j])
                else:
                    # Use encoder feature
                    feature_set.append(enc_proj[j])

            # Target spatial size for KV alignment = same as deepest feature
            target_h = enc_proj[N - 1].shape[2]
            target_w = enc_proj[N - 1].shape[3]

            # Align and mix features → Xkv
            x_kv = self.feature_aligners[i](feature_set, target_h, target_w)

            # Decoder stage
            dec_out = self.decoder_stages[i](x_q_flat, x_kv, Hq, Wq)
            dec_outputs[enc_idx] = rearrange(
                dec_out, "b (h w) c -> b c h w", h=Hq, w=Wq)

        # Upsample all decoder outputs to match the shallowest (largest)
        target_h = enc_proj[0].shape[2]
        target_w = enc_proj[0].shape[3]
        upsampled = []
        for d in dec_outputs:
            if d.shape[2] != target_h or d.shape[3] != target_w:
                d = F.interpolate(d, size=(target_h, target_w),
                                  mode="bilinear", align_corners=False)
            upsampled.append(d)

        # Concatenate and predict
        fused = torch.cat(upsampled, dim=1)  # (B, N*decoder_dim, H/4, W/4)
        logits = self.final_fuse(fused)      # (B, num_classes, H/4, W/4)
        return logits
