"""SLOT 1 (patch GNN) and SLOT 2 (token encoder)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv, GINConv, GCNConv, SAGEConv, GATConv, GATv2Conv, TransformerConv,
)

from .config import EDGE_CONVS


class CategoricalEncoder(nn.Module):
    """Sum of per-column embeddings. x holds integer CATEGORY INDICES, not magnitudes --
    feeding it to nn.Linear 'works' and quietly destroys the signal."""

    def __init__(self, dims, dim):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(d, dim) for d in dims])
        for e in self.emb:
            nn.init.xavier_uniform_(e.weight)

    def forward(self, x):
        return sum(self.emb[i](x[:, i]) for i in range(len(self.emb)))


def AtomEncoder(dim):
    # ogb is imported here rather than at module level: only the molecular fallback needs
    # it, and the protein project should not force students to install it.
    from ogb.utils.features import get_atom_feature_dims
    return CategoricalEncoder(get_atom_feature_dims(), dim)


def BondEncoder(dim):
    from ogb.utils.features import get_bond_feature_dims
    return CategoricalEncoder(get_bond_feature_dims(), dim)


class ResidueEncoder(nn.Module):
    """20 amino acids + UNK(20) + MASK(21).

    The MASK row exists even for models that never emit it (graph-level JEPA), so that every
    protein encoder has an identical parameter shape and the graph-JEPA / node-JEPA / scratch
    arms are comparable parameter-for-parameter.
    """

    def __init__(self, dim, n_aa=22):
        super().__init__()
        self.emb = nn.Embedding(n_aa, dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x):
        return self.emb(x[:, 0])


class GeometricEdgeEncoder(nn.Module):
    """Edge = (CA-CA distance, sequence separation).

    Distance goes through a radial basis expansion rather than straight into a Linear: a
    raw scalar distance forces the network to learn a sharp non-linearity from one input,
    while RBFs hand it a smooth local basis. Standard in every structural protein GNN.
    """

    def __init__(self, dim, n_rbf=16, cutoff=20.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.width = cutoff / n_rbf
        self.proj = nn.Linear(n_rbf + 1, dim)

    def forward(self, e):
        d, sep = e[:, :1], e[:, 1:2]
        rbf = torch.exp(-((d - self.centers) ** 2) / (2 * self.width ** 2))
        return self.proj(torch.cat([rbf, torch.log1p(sep)], -1))


def make_input_encoders(cfg):
    if cfg.domain == "protein":
        return ResidueEncoder(cfg.dim), GeometricEdgeEncoder(cfg.dim)
    return AtomEncoder(cfg.dim), BondEncoder(cfg.dim)


def make_conv(conv_type, dim, heads=4, sage_aggr="mean"):
    mlp = lambda: nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))
    if conv_type == "GINEConv":
        return GINEConv(mlp(), train_eps=True)
    if conv_type == "GINConv":
        return GINConv(mlp(), train_eps=True)
    if conv_type == "GCNConv":
        return GCNConv(dim, dim)
    if conv_type == "SAGEConv":
        return SAGEConv(dim, dim, aggr=sage_aggr)
    if conv_type == "GATConv":
        return GATConv(dim, dim // heads, heads=heads, edge_dim=dim)
    if conv_type == "GATv2Conv":
        return GATv2Conv(dim, dim // heads, heads=heads, edge_dim=dim)
    if conv_type == "TransformerConv":
        return TransformerConv(dim, dim // heads, heads=heads, edge_dim=dim)
    raise ValueError(f"unknown conv_type {conv_type}")


class PatchGNN(nn.Module):
    """Message passing INSIDE each patch. Convs outside EDGE_CONVS drop bond features."""

    def __init__(self, cfg):
        super().__init__()
        self.uses_edges = cfg.conv_type in EDGE_CONVS
        self.atom, bond = make_input_encoders(cfg)
        self.bond = bond if self.uses_edges else None
        self.convs = nn.ModuleList(
            [make_conv(cfg.conv_type, cfg.dim, cfg.n_heads, cfg.sage_aggr)
             for _ in range(cfg.n_gnn_layers)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(cfg.dim) for _ in range(cfg.n_gnn_layers)])
        self.dropout = cfg.dropout

    def forward(self, x, edge_index, edge_attr):
        h = self.atom(x)
        e = self.bond(edge_attr) if self.uses_edges else None
        for conv, norm in zip(self.convs, self.norms):
            m = conv(h, edge_index, e) if self.uses_edges else conv(h, edge_index)
            h = h + F.dropout(F.relu(norm(m)), self.dropout, self.training)
        return h


class MLPMixerBlock(nn.Module):
    def __init__(self, dim, n_patches, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.tok = nn.Sequential(nn.Linear(n_patches, n_patches), nn.GELU(), nn.Linear(n_patches, n_patches))
        self.ch = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))

    def forward(self, z, pad):
        z = z + self.tok(self.n1(z).transpose(1, 2)).transpose(1, 2)
        return z + self.ch(self.n2(z))


class TokenEncoder(nn.Module):
    """Operates on the patch-token sequence [B, P, D].

    NOTE (deviation from the reference repo): context and target encoders share this one
    architecture so the EMA parameter-for-parameter copy is well defined. The reference
    builds them from two different classes yet still zips their parameter lists.
    """

    def __init__(self, cfg):
        super().__init__()
        self.kind = cfg.token_encoder
        d = cfg.dim
        if self.kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg.n_heads, dim_feedforward=2 * d,
                dropout=cfg.dropout, batch_first=True, norm_first=True,
            )
            self.net = nn.TransformerEncoder(layer, num_layers=cfg.n_token_layers)
        elif self.kind == "mixer":
            self.net = nn.ModuleList(
                [MLPMixerBlock(d, cfg.n_patches, cfg.dropout) for _ in range(cfg.n_token_layers)]
            )
        elif self.kind == "mlp":  # per-token, no mixing -- the reference "Standard" encoder
            self.net = nn.Sequential(*[
                m for _ in range(cfg.n_token_layers)
                for m in (nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            ])
        else:
            raise ValueError(f"unknown token_encoder {cfg.token_encoder}")

    def forward(self, z, valid=None):
        # NO final LayerNorm here, deliberately. LayerNorm zero-means the feature dim, which
        # makes the hyperbolic angle z.mean(-1) identically 0 -> every target becomes the
        # constant (1,0), loss -> 0, and the encoder learns nothing. Target normalisation
        # belongs in GraphJEPA.to_geometry, where it can be geometry-aware.
        pad = ~valid if valid is not None else None
        if self.kind == "transformer":
            return self.net(z, src_key_padding_mask=pad)
        if self.kind == "mixer":
            for blk in self.net:
                z = blk(z, pad)
            return z
        return self.net(z)
