from dataclasses import dataclass, asdict

import torch


@dataclass
class Config:
    # --- domain ---
    domain: str = "mol"              # mol | protein  (selects node/edge input encoders)

    # --- data ---
    n_molecules: int = 50_000
    data_root: str = "data"

    # --- tokenizer ---
    patcher: str = "random"          # random | khop | spatial
    n_patches: int = 8
    # Residue kNN graphs are dense: 1-hop expansion from ~17 seed residues can swallow the
    # whole protein, making the context patch ~= the entire graph and the task trivial.
    # Proteins default to 0 hops (see protein_config()).
    num_hops: int = 1

    # --- SLOT 1: patch GNN ---
    conv_type: str = "GINEConv"      # GINEConv|GCNConv|SAGEConv|GATv2Conv|TransformerConv|GINConv
    sage_aggr: str = "mean"          # mean | max   (SAGEConv only)
    n_gnn_layers: int = 2

    # --- SLOT 2: token encoder ---
    token_encoder: str = "transformer"   # transformer | mlp | mean
    n_token_layers: int = 4
    n_heads: int = 4

    # --- SLOT 3: predictor ---
    predictor_layers: int = 3
    # How the target's position code reaches the predictor. The reference repo ADDs it to
    # the context embedding; Polymer-JEPA concatenates. Adding a d-dim code to a d-dim
    # embedding can swamp the context signal.
    predictor_conditioning: str = "add"      # add | concat

    # --- SLOT 4: target geometry ---
    target_geometry: str = "hyperbola"   # hyperbola | euclidean | poincare

    # --- JEPA ---
    n_context: int = 1
    n_targets: int = 4
    use_ema: bool = True
    ema_start: float = 0.996
    ema_end: float = 1.0
    target_layernorm: bool = True

    # --- positional conditioning ---
    use_rwse: bool = True
    rwse_dim: int = 16

    # --- model ---
    dim: int = 128
    dropout: float = 0.0

    # --- optimisation ---
    epochs: int = 15
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.0
    seed: int = 0
    # "auto" resolves to cuda when a GPU is visible and cpu otherwise. Colab starts on a
    # CPU runtime by default, so hardcoding "cuda" here would crash for anyone who forgot
    # Runtime > Change runtime type > T4 GPU.
    device: str = "auto"
    num_workers: int = 0

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dim % self.n_heads:
            raise ValueError(f"dim {self.dim} not divisible by n_heads {self.n_heads}")
        if self.n_context + self.n_targets > self.n_patches:
            raise ValueError(
                f"n_context+n_targets ({self.n_context}+{self.n_targets}) > n_patches ({self.n_patches})"
            )

    def to_dict(self):
        return asdict(self)


def protein_config(**kw):
    """Defaults tuned for residue graphs rather than molecular graphs."""
    base = dict(domain="protein", patcher="spatial", num_hops=0, n_patches=8,
                n_context=2, n_targets=4, conv_type="GINEConv", dim=128,
                batch_size=64, epochs=20, ema_start=0.99, ema_end=0.9999)
    base.update(kw)
    return Config(**base)


# Convs that consume multi-dimensional edge features. Anything not listed here
# silently DROPS bond chemistry -- a confound that must be controlled for.
EDGE_CONVS = {"GINEConv", "GATConv", "GATv2Conv", "TransformerConv", "GENConv", "ResGatedGraphConv"}

# Neighbour aggregator per conv. This is the axis the project's central hypothesis rides on.
AGGREGATOR = {
    "GINEConv": "sum",
    "GINConv": "sum",
    "SAGEConv": "mean/max",
    "GCNConv": "mean",
    "GATv2Conv": "weighted-mean",
    "GATConv": "weighted-mean",
    "TransformerConv": "weighted-mean",
}
