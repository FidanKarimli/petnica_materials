"""Graph-JEPA: predict the EMBEDDINGS of held-out patches from one visible patch."""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from .encoders import PatchGNN, TokenEncoder

EPS = 1e-5


def mlp(nin, nout, layers, hidden=None):
    hidden = hidden or nin
    if layers <= 1:
        return nn.Linear(nin, nout)
    mods = [nn.Linear(nin, hidden), nn.ReLU()]
    for _ in range(layers - 2):
        mods += [nn.Linear(hidden, hidden), nn.ReLU()]
    return nn.Sequential(*mods, nn.Linear(hidden, nout))


def to_ball(z):
    n = z.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return z * (torch.tanh(n) / n) * (1 - EPS)


def poincare_dist(u, v):
    du = (1 - u.pow(2).sum(-1)).clamp(min=EPS)
    dv = (1 - v.pow(2).sum(-1)).clamp(min=EPS)
    x = 1 + 2 * (u - v).pow(2).sum(-1) / (du * dv)
    return torch.acosh(x.clamp(min=1 + EPS))


def sample_context_targets(valid, n_ctx, n_tgt):
    """Disjoint by construction: two slices of one random permutation, valid patches first."""
    score = torch.rand(valid.shape, device=valid.device).masked_fill(~valid, -1.0)
    order = score.argsort(dim=1, descending=True)
    ctx, tgt = order[:, :n_ctx], order[:, n_ctx:n_ctx + n_tgt]
    return ctx, tgt, valid.gather(1, tgt)


def collapse_stats(z):
    """Loss going to zero WITH variance going to zero is collapse. Track both."""
    std = z.std(0).mean().item()
    zc = z - z.mean(0)
    cov = zc.T @ zc / max(len(z) - 1, 1)
    ev = torch.linalg.eigvalsh(cov.float()).clamp(min=0)
    p = ev / ev.sum().clamp(min=EPS)
    eff_rank = torch.exp(-(p * (p + EPS).log()).sum()).item()
    return std, eff_rank


class GraphJEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_gnn = PatchGNN(cfg)                       # shared, gradient-trained
        self.pe_proj = nn.Linear(cfg.rwse_dim, cfg.dim) if cfg.use_rwse else None
        self.context_encoder = TokenEncoder(cfg)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)            # EMA-only, never optimised
        out_dim = cfg.dim if cfg.target_geometry == "euclidean" else 2
        in_dim = cfg.dim * (2 if cfg.predictor_conditioning == "concat" else 1)
        self.predictor = mlp(in_dim, out_dim, cfg.predictor_layers, hidden=cfg.dim)

    # ---- patch tokens ------------------------------------------------------
    def embed_patches(self, batch):
        b, p = batch.num_graphs, self.cfg.n_patches
        h = self.patch_gnn(
            batch.x[batch.subgraphs_nodes_mapper],
            batch.combined_subgraphs,
            batch.edge_attr[batch.subgraphs_edges_mapper],
        )
        idx, n_tok = batch.subgraphs_batch, b * p
        s = scatter(h, idx, dim=0, dim_size=n_tok, reduce="sum")
        c = scatter(torch.ones_like(idx, dtype=h.dtype), idx, dim=0, dim_size=n_tok, reduce="sum")
        tok = (s / c.clamp(min=1).unsqueeze(-1)).view(b, p, -1)
        valid = batch.patch_valid.view(b, p)

        if self.pe_proj is None:
            pe = torch.zeros_like(tok)
        else:
            pn = scatter(batch.rwse[batch.subgraphs_nodes_mapper], idx, dim=0,
                         dim_size=n_tok, reduce="max")       # Eq.2: elementwise max over patch
            pe = self.pe_proj(torch.nan_to_num(pn, neginf=0.0, posinf=0.0)).view(b, p, -1)
        return tok, pe, valid

    def to_geometry(self, z):
        """Map target-encoder output to the space the predictor regresses in.

        Normalisation is geometry-aware on purpose: I-JEPA LayerNorms its targets for
        scale stability, but doing that under `hyperbola` zeroes the angle z.mean(-1) and
        the task degenerates to predicting the constant (1,0).
        """
        g = self.cfg.target_geometry
        if g == "euclidean":
            return F.layer_norm(z, z.shape[-1:]) if self.cfg.target_layernorm else z
        if g == "hyperbola":                                  # 2D unit hyperbola x^2-y^2=1
            a = z.mean(-1, keepdim=True)
            return torch.cat([torch.cosh(a), torch.sinh(a)], -1)
        if g == "poincare":
            return to_ball(z[..., :2])
        raise ValueError(g)

    @torch.no_grad()
    def target_spread(self, batch):
        """Std of the regression targets across patches. ~0 means the objective is
        degenerate: the predictor can win by emitting a constant."""
        tok, _, valid = self.embed_patches(batch)
        y = self.to_geometry(self.target_encoder(tok, valid))
        return y[valid].std(0).mean().item()

    @staticmethod
    def _gather(t, idx):
        return t.gather(1, idx.unsqueeze(-1).expand(-1, -1, t.size(-1)))

    # ---- training ----------------------------------------------------------
    def forward(self, batch):
        tok, pe, valid = self.embed_patches(batch)
        ctx_i, tgt_i, tgt_ok = sample_context_targets(valid, self.cfg.n_context, self.cfg.n_targets)

        # Masked mean: with n_context>1 a small molecule can draw empty patches, and
        # averaging their embeddings in would dilute the context with zeros.
        ctx_ok = valid.gather(1, ctx_i)
        h_ctx = self.context_encoder(self._gather(tok, ctx_i), ctx_ok)
        w = ctx_ok.unsqueeze(-1).to(h_ctx.dtype)
        z_ctx = (h_ctx * w).sum(1) / w.sum(1).clamp(min=1)

        with torch.no_grad():                                 # stop-grad on the target branch
            z_all = self.target_encoder(tok.detach(), valid)
            y = self.to_geometry(self._gather(z_all, tgt_i))

        # The predictor must be told WHICH patch it is predicting; without the position code
        # it can only regress to the mean of all targets -- a soft collapse that still shows
        # a healthy-looking loss curve.
        c = z_ctx.unsqueeze(1).expand(-1, tgt_i.size(1), -1)
        p = self._gather(pe, tgt_i)
        pred = self.predictor(torch.cat([c, p], -1)
                              if self.cfg.predictor_conditioning == "concat" else c + p)

        if self.cfg.target_geometry == "poincare":
            per = poincare_dist(to_ball(pred), y)
        else:
            per = F.smooth_l1_loss(pred, y, reduction="none").mean(-1)
        return (per * tgt_ok).sum() / tgt_ok.sum().clamp(min=1)

    @torch.no_grad()
    def ema_update(self, m):
        """theta_target <- m*theta_target + (1-m)*theta_context. m=0 => weight sharing."""
        if not self.cfg.use_ema:
            m = 0.0
        for q, k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            if q.shape != k.shape:
                raise RuntimeError(f"EMA shape mismatch {q.shape} vs {k.shape}")
            k.data.mul_(m).add_((1.0 - m) * q.detach().data)
        for bq, bk in zip(self.context_encoder.buffers(), self.target_encoder.buffers()):
            bk.data.copy_(bq.data)

    # ---- evaluation --------------------------------------------------------
    @torch.no_grad()
    def encode(self, batch):
        """Frozen graph representation: all patches -> target encoder -> masked mean."""
        tok, _, valid = self.embed_patches(batch)
        z = self.target_encoder(tok, valid)
        w = valid.unsqueeze(-1).to(z.dtype)
        return (z * w).sum(1) / w.sum(1).clamp(min=1)
