"""RecAST: Reconstruction-Assisted Spatio-Temporal transformer.

The model receives a causally filled input window, calendar features and the
availability mask M. Each of L blocks applies temporal self-attention and a
mask-conditioned spatial layer. A linear layer maps the final representation to
the forecast. A training-only linear head reconstructs input values.

Tensor layout: (batch, time, node, channel).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_adjacency(adj):
    """Return D^{-1/2} A_bar D^{-1/2}, where A_bar is the binarized adjacency with
    self-loops and D holds the row (out-)degrees of A_bar.

    For a directed source graph the result is not symmetric in general.
    """
    adj = (np.asarray(adj) > 0).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(adj.sum(axis=1))  # degrees are >= 1 because of the self-loops
    return torch.from_numpy(adj * d_inv_sqrt[:, None] * d_inv_sqrt[None, :])


class AttentionLayer(nn.Module):
    """Multi-head scaled dot-product attention across dimension -2."""

    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.head_dim = model_dim // num_heads
        self.fc_q = nn.Linear(model_dim, model_dim)
        self.fc_k = nn.Linear(model_dim, model_dim)
        self.fc_v = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        query = torch.cat(torch.split(self.fc_q(query), self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(self.fc_k(key), self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(self.fc_v(value), self.head_dim, dim=-1), dim=0)
        score = torch.softmax((query @ key.transpose(-1, -2)) / self.head_dim ** 0.5, dim=-1)
        out = torch.cat(torch.split(score @ value, batch_size, dim=0), dim=-1)
        return self.out_proj(out)


class TemporalSelfAttention(nn.Module):
    """Self-attention over the input steps of each sensor, then a feed-forward network."""

    def __init__(self, model_dim, feed_forward_dim, num_heads, dropout):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        xt = x.transpose(1, 2)  # (B, N, T, D)
        out = self.ln1(xt + self.dropout1(self.attn(xt, xt, xt)))
        out = self.ln2(out + self.dropout2(self.feed_forward(out)))
        return out.transpose(1, 2)


class MaskConditionedSpatialLayer(nn.Module):
    """Graph-convolution and spatial-attention updates weighted by a sigmoid gate.

    g = A_hat h W_f + A_adp h W_a, with A_adp = softmax(ReLU(E1 E2^T)).
    s = multi-head attention across sensors.
    gamma = sigmoid(W_gamma [h || g || s || M || A_hat M]).
    h' = LN(h + Drop(gamma_1 g + gamma_2 s)), followed by a feed-forward sub-layer.

    The two gate values are independent coefficients and need not sum to one.
    A_hat M is a graph-weighted availability statistic, not a fraction of observed
    neighbors, because A_hat is not row normalized.
    """

    def __init__(self, model_dim, feed_forward_dim, num_heads, dropout,
                 num_nodes, adp_emb_dim, adj_norm, mask_dim=1):
        super().__init__()
        self.register_buffer('adj_norm', adj_norm)
        self.e1 = nn.Parameter(torch.empty(num_nodes, adp_emb_dim))
        self.e2 = nn.Parameter(torch.empty(num_nodes, adp_emb_dim))
        nn.init.xavier_uniform_(self.e1)
        nn.init.xavier_uniform_(self.e2)
        self.w_adp = nn.Linear(model_dim, model_dim)
        self.w_fixed = nn.Linear(model_dim, model_dim)
        self.attn = AttentionLayer(model_dim, num_heads)
        self.gate = nn.Linear(3 * model_dim + 2 * mask_dim, 2)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: (B, T, N, D); mask: (B, T, N, mask_dim), 1 = available reading
        gcn = torch.einsum('uv,btvd->btud', self.adj_norm, self.w_fixed(x))
        adp = torch.softmax(F.relu(self.e1 @ self.e2.t()), dim=-1)
        gcn = gcn + torch.einsum('uv,btvd->btud', adp, self.w_adp(x))
        s_attn = self.attn(x, x, x)
        nbr = torch.einsum('uv,btvd->btud', self.adj_norm, mask)
        g = torch.sigmoid(self.gate(torch.cat([x, gcn, s_attn, mask, nbr], dim=-1)))
        spatial = g[..., 0:1] * gcn + g[..., 1:2] * s_attn
        out = self.ln1(x + self.dropout1(spatial))
        out = self.ln2(out + self.dropout2(self.feed_forward(out)))
        return out


class RecAST(nn.Module):
    """Reconstruction-Assisted Spatio-Temporal transformer.

    Args:
        num_nodes: number of sensors N.
        adj: (N, N) adjacency of the sensor graph (weights are binarized).
        in_steps, out_steps: history length P and horizon Q (12 and 12 in the paper).
        steps_per_day: 288 for 5-min data.
        Embedding widths default to the paper: 24 (value), 24 (time of day),
        24 (day of week), 80 (node), 8 (mask), so D = 160.
        reconstruction: build the training-only reconstruction head.

    forward(x, mask):
        x: (B, P, N, 3) with channels [filled standardized value, time of day in
           [0, 1), day of week in {0..6}].
        mask: (B, P, N, 1) availability mask M (bool or float, 1 = available).
        Returns the forecast (B, Q, N, 1) in standardized units, and additionally
        the reconstruction (B, P, N, 1) when return_reconstruction=True.
    """

    def __init__(self, num_nodes, adj, in_steps=12, out_steps=12, steps_per_day=288,
                 input_embedding_dim=24, tod_embedding_dim=24, dow_embedding_dim=24,
                 node_embedding_dim=80, mask_embedding_dim=8, feed_forward_dim=256,
                 num_heads=4, num_layers=3, dropout=0.1, adp_emb_dim=16,
                 reconstruction=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.model_dim = (input_embedding_dim + tod_embedding_dim + dow_embedding_dim
                          + node_embedding_dim + mask_embedding_dim)
        if self.model_dim % num_heads:
            raise ValueError('Model width must be divisible by the number of heads')

        self.input_proj = nn.Linear(1, input_embedding_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, node_embedding_dim))
        nn.init.xavier_uniform_(self.node_embedding)
        self.mask_proj = nn.Linear(1, mask_embedding_dim)

        adj_norm = normalized_adjacency(adj)
        self.temporal_layers = nn.ModuleList([
            TemporalSelfAttention(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)])
        self.spatial_layers = nn.ModuleList([
            MaskConditionedSpatialLayer(self.model_dim, feed_forward_dim, num_heads, dropout,
                                        num_nodes, adp_emb_dim, adj_norm.clone())
            for _ in range(num_layers)])
        self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps)
        self.recon_proj = nn.Linear(self.model_dim, 1) if reconstruction else None

    def encode(self, x, mask):
        mask = mask.to(x.dtype)
        batch_size = x.shape[0]
        tod = (x[..., 1] * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)
        dow = x[..., 2].long().clamp(0, 6)
        h = torch.cat([
            self.input_proj(x[..., :1]),
            self.tod_embedding(tod),
            self.dow_embedding(dow),
            self.node_embedding.expand(batch_size, self.in_steps, *self.node_embedding.shape),
            self.mask_proj(mask),
        ], dim=-1)
        for temporal, spatial in zip(self.temporal_layers, self.spatial_layers):
            h = spatial(temporal(h), mask)
        return h

    def project(self, h):
        batch_size = h.shape[0]
        out = h.transpose(1, 2).reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
        out = self.output_proj(out).view(batch_size, self.num_nodes, self.out_steps, 1)
        return out.transpose(1, 2)

    def forward(self, x, mask, return_reconstruction=False):
        h = self.encode(x, mask)
        forecast = self.project(h)
        if not return_reconstruction:
            return forecast
        if self.recon_proj is None:
            raise ValueError('Model was built without the reconstruction head')
        return forecast, self.recon_proj(h)
