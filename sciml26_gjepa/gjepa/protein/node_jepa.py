"""Node-level JEPA for residue graphs.

WHY THIS EXISTS. Graph-JEPA is graph-level: a patch GNN feeds a Transformer over patch
tokens, and the EMA/stop-grad machinery lives in that token encoder. But ddG is a per-residue
question, so downstream we can only reuse the patch GNN -- roughly 200k of the model's
parameters -- and we discard the token encoder where most of the pretraining capacity sits.
I-JEPA transfers its *big* encoder; the graph-level port transfers the small one.

Node-JEPA fixes the mismatch: the encoder IS the residue-level GNN, so 100% of what was
pretrained is what gets fine-tuned.

THE TASK. Replace the amino-acid identity of a subset of residues with a MASK token, leaving
the structure (edges, distances, coordinates) fully visible. Predict the *latent*
representations of those residues, as produced by an EMA copy of the encoder reading the
unmasked graph.

That is deliberately the ProteinMPNN-shaped question -- "what belongs at this position given
its structural neighbourhood" -- but answered in embedding space rather than by classifying
the residue. Pythia (Cell Innovation 2024) does the input-space version for ddG; the
latent-space version is the open comparison this project can actually run.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders import PatchGNN
from ..jepa import mlp, collapse_stats
from loguru import logger

MASK_IDX = 21          # 20 amino acids + UNK(20) + MASK(21)
N_AA_MASKED = 22


class NodeJEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = PatchGNN(cfg)
        # ResidueEncoder already carries the MASK row (n_aa=22) for every protein model, so
        # graph-JEPA / node-JEPA / scratch encoders are identical parameter-for-parameter.
        assert self.encoder.atom.emb.num_embeddings == N_AA_MASKED, \
            "residue embedding table has no MASK row"

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.requires_grad_(False)
        self.predictor = mlp(cfg.dim, cfg.dim, cfg.predictor_layers, hidden=cfg.dim)
        self.mask_ratio = getattr(cfg, "mask_ratio", 0.25)
        self.mask_mode = getattr(cfg, "mask_mode", "region")   # region | random

    # ---- pretraining -------------------------------------------------------
    def sample_mask(self, batch):
        """Which residues to hide.

        'region' masks a contiguous 3D neighbourhood -- a piece of the fold -- so the task is
        "reconstruct this structural region from the rest". 'random' scatters masked residues,
        which is easier because each one keeps unmasked sequence neighbours.
        """
        x = batch.x
        if self.mask_mode == "random" or getattr(batch, "pos", None) is None:
            return torch.rand(x.size(0), device=x.device) < self.mask_ratio
        m = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        for g in range(batch.num_graphs):
            lo, hi = int(batch.ptr[g]), int(batch.ptr[g + 1])
            n = hi - lo
            if n < 2:
                continue
            pos = batch.pos[lo:hi]
            seed = pos[torch.randint(n, (1,), device=pos.device)]
            k = max(1, int(round(self.mask_ratio * n)))
            near = torch.cdist(seed, pos)[0].topk(k, largest=False).indices
            m[lo + near] = True
        return m

    def forward(self, batch):
        x, ei, ea = batch.x, batch.edge_index, batch.edge_attr
        m = self.sample_mask(batch)
        if not m.any():                                   # degenerate tiny batch
            m[torch.randint(x.size(0), (1,), device=x.device)] = True

        x_masked = x.clone()
        x_masked[m, 0] = MASK_IDX
        h_ctx = self.encoder(x_masked, ei, ea)

        with torch.no_grad():                             # stop-grad on the target branch
            h_tgt = self.target_encoder(x, ei, ea).detach()

        return F.smooth_l1_loss(self.predictor(h_ctx[m]), h_tgt[m])

    @torch.no_grad()
    def ema_update(self, mom):
        if not self.cfg.use_ema:
            mom = 0.0
        for q, k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            k.data.mul_(mom).add_((1.0 - mom) * q.detach().data)
        for bq, bk in zip(self.encoder.buffers(), self.target_encoder.buffers()):
            bk.data.copy_(bq.data)

    @torch.no_grad()
    def target_spread(self, batch):
        h = self.target_encoder(batch.x, batch.edge_index, batch.edge_attr)
        return h.std(0).mean().item()

    # ---- evaluation --------------------------------------------------------
    @torch.no_grad()
    def encode(self, batch):
        """Graph-level pooled embedding, for collapse diagnostics only."""
        from torch_geometric.utils import scatter

        h = self.target_encoder(batch.x, batch.edge_index, batch.edge_attr)
        return scatter(h, batch.batch, dim=0, dim_size=batch.num_graphs, reduce="mean")

    @property
    def patch_gnn(self):
        """Downstream code transfers `.patch_gnn`; here that is the whole encoder."""
        return self.encoder


def pretrain_node_jepa(cfg, data_list, verbose=True):
    """No patching needed -- the objective is already per-node, so plain batching suffices."""
    import time
    import numpy as np
    import pandas as pd
    from torch_geometric.loader import DataLoader

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = cfg.device if torch.cuda.is_available() else "cpu"
    model = NodeJEPA(cfg).to(dev)
    loader = DataLoader(data_list, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = max(len(loader) * cfg.epochs, 1)
    ema = (cfg.ema_start + i * (cfg.ema_end - cfg.ema_start) / total for i in range(total + 1))

    hist, t0 = [], time.time()
    for ep in range(cfg.epochs):
        model.train()
        tot = n = 0
        for batch in loader:
            batch = batch.to(dev)
            loss = model(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.ema_update(next(ema))
            tot += loss.item() * batch.num_graphs
            n += batch.num_graphs
        model.eval()
        with torch.no_grad():
            z = torch.cat([model.encode(b.to(dev)) for i, b in enumerate(loader) if i < 4])
        std, rank = collapse_stats(z)
        hist.append(dict(epoch=ep, loss=tot / max(n, 1), emb_std=std, eff_rank=rank,
                         secs=time.time() - t0))
        if verbose:
            logger.info(f"  ep {ep:>3}  loss {hist[-1]['loss']:.4f}  emb_std {std:.4f}  "
                  f"eff_rank {rank:6.2f}  {hist[-1]['secs']:.0f}s")
        if std < 1e-4:
            logger.warning("  !! COLLAPSE: node representations are constant")
            break
    return model, pd.DataFrame(hist)
