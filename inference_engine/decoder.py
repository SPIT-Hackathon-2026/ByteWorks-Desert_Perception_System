"""
U-MixFormer Decoder implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MixAttention(nn.Module):
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
        B, Nq, C = x_q.shape
        q = rearrange(self.q_proj(x_q), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.k_proj(x_kv), "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.v_proj(x_kv), "b n (h d) -> b h n d", h=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = rearrange(attn @ v, "b h n d -> b n (h d)")
        out = self.proj_drop(self.out_proj(out))
        return out

class MixFFN(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, drop: float = 0.0):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.dwconv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.drop(self.act(x))
        x = self.drop(self.fc2(x))
        return x

class DecoderStage(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, ffn_expansion: int = 4,
                 attn_drop: float = 0.0, ffn_drop: float = 0.1, depth: int = 1):
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

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x_q
        for blk in self.blocks:
            x = blk["mix_attn"](blk["norm_q"](x), blk["norm_kv"](x_kv)) + x
            x = blk["ffn"](blk["norm_ff"](x), H, W) + x
        return x

class FeatureAligner(nn.Module):
    def __init__(self, in_channels_list: list[int], out_channels: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(c), nn.Linear(c, out_channels))
            for c in in_channels_list
        ])

    def forward(self, features: list[torch.Tensor], target_h: int, target_w: int) -> torch.Tensor:
        aligned = []
        for feat, proj in zip(features, self.projections):
            if feat.shape[2] != target_h or feat.shape[3] != target_w:
                feat = F.adaptive_avg_pool2d(feat, (target_h, target_w))
            feat = rearrange(feat, "b c h w -> b (h w) c")
            aligned.append(proj(feat))
        return torch.cat(aligned, dim=1)

class UMixFormerDecoder(nn.Module):
    def __init__(self, encoder_channels: list[int], decoder_dim: int = 256,
                 num_heads: int = 8, ffn_expansion: int = 4, attn_drop: float = 0.0,
                 ffn_drop: float = 0.1, decoder_depth: int = 1, num_classes: int = 4):
        super().__init__()
        self.N = len(encoder_channels)
        self.encoder_projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, decoder_dim, 1), nn.BatchNorm2d(decoder_dim), nn.GELU())
            for c in encoder_channels
        ])
        self.feature_aligners = nn.ModuleList([
            FeatureAligner([decoder_dim] * self.N, decoder_dim) for _ in range(self.N)
        ])
        self.decoder_stages = nn.ModuleList([
            DecoderStage(decoder_dim, num_heads, ffn_expansion, attn_drop, ffn_drop, decoder_depth)
            for _ in range(self.N)
        ])
        self.final_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * self.N, decoder_dim, 1),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        enc_proj = [proj(feat) for proj, feat in zip(self.encoder_projections, encoder_features)]
        dec_outputs = [None] * self.N
        target_h, target_w = enc_proj[-1].shape[2], enc_proj[-1].shape[3]

        for i in range(self.N):
            enc_idx = self.N - 1 - i
            x_q = enc_proj[enc_idx]
            Hq, Wq = x_q.shape[2], x_q.shape[3]
            x_q_flat = rearrange(x_q, "b c h w -> b (h w) c")

            feature_set = []
            for j in range(self.N):
                if j > enc_idx and dec_outputs[j] is not None:
                    feature_set.append(dec_outputs[j])
                else:
                    feature_set.append(enc_proj[j])

            x_kv = self.feature_aligners[i](feature_set, target_h, target_w)
            dec_out = self.decoder_stages[i](x_q_flat, x_kv, Hq, Wq)
            dec_outputs[enc_idx] = rearrange(dec_out, "b (h w) c -> b c h w", h=Hq, w=Wq)

        target_h, target_w = enc_proj[0].shape[2], enc_proj[0].shape[3]
        upsampled = [F.interpolate(d, size=(target_h, target_w), mode="bilinear", align_corners=False)
                     if d.shape[2] != target_h or d.shape[3] != target_w else d for d in dec_outputs]
        return self.final_fuse(torch.cat(upsampled, dim=1))
