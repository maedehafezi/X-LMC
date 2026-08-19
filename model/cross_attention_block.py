import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """
    One directional cross-attention block:
    query comes from x_q
    key/value come from x_kv

    Input:
        x_q  : [B, Nq, D]
        x_kv : [B, Nk, D]

    Output:
        out  : [B, Nq, D]
    """
    def __init__(self, dim=768, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm_mlp = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x_q, x_kv):
        # Cross-attention
        q = self.norm_q(x_q)
        kv = self.norm_kv(x_kv)

        attn_out, attn_weights = self.attn(
            query=q,
            key=kv,
            value=kv
        )

        # Residual connection after attention
        x = x_q + attn_out

        # Feed-forward / MLP block
        x = x + self.mlp(self.norm_mlp(x))

        return x, attn_weights


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention:
    A attends to B
    B attends to A

    Input:
        tokens_a : [B, N, D]
        tokens_b : [B, N, D]

    Output:
        a_out    : [B, N, D]
        b_out    : [B, N, D]
    """
    def __init__(self, dim=768, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.a_to_b = CrossAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        self.b_to_a = CrossAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

    def forward(self, tokens_a, tokens_b):
        a_out, attn_a_to_b = self.a_to_b(tokens_a, tokens_b)
        b_out, attn_b_to_a = self.b_to_a(tokens_b, tokens_a)

        return a_out, b_out, attn_a_to_b, attn_b_to_a