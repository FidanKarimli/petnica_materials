"""ddG fine-tuning: per-residue readout from the pretrained encoder.

Design note. Graph-JEPA is a GRAPH-level method, but ddG is a per-residue question, so the
readout cannot be the pooled graph embedding. What transfers is the PATCH GNN, applied here
to the whole residue graph (no patching) to give one embedding per residue. The token
encoder is pretext-task scaffolding, exactly as I-JEPA's predictor is discarded downstream.

Efficiency. All ~900 mutations of a domain share one wild-type structure, so we batch BY
PROTEIN: encode a handful of structures, then score all of their mutations from the cached
residue embeddings. That turns 271k training examples into ~239 structure forward passes
per epoch.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from torch_geometric.data import Batch

from .data import AA_IDX, N_AA, UNK
from loguru import logger


def aa_idx(series):
    return torch.tensor([AA_IDX.get(a, UNK) for a in series], dtype=torch.long)


class DDGHead(nn.Module):
    """(residue embedding, wt aa, mutant aa) -> ddG."""

    def __init__(self, dim, hidden=256, dropout=0.1):
        super().__init__()
        self.wt = nn.Embedding(N_AA, dim)
        self.mut = nn.Embedding(N_AA, dim)
        self.net = nn.Sequential(
            nn.Linear(3 * dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, h, wt, mut):
        return self.net(torch.cat([h, self.wt(wt), self.mut(mut)], -1)).squeeze(-1)


def node_embeddings(model, batch):
    """Pretrained patch GNN over the FULL graph -> [n_nodes, dim]."""
    return model.patch_gnn(batch.x, batch.edge_index, batch.edge_attr)


class ProteinBatches:
    """Yields (Batch of structures, row-index array) grouped by protein."""

    def __init__(self, structures, df, n_proteins=8, shuffle=True, seed=0):
        self.s, self.df, self.k, self.shuffle = structures, df, n_proteins, shuffle
        self.names = sorted(df.WT_name.unique())
        self.by_name = {n: np.flatnonzero((df.WT_name == n).values) for n in self.names}
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return int(np.ceil(len(self.names) / self.k))

    def __iter__(self):
        names = list(self.names)
        if self.shuffle:
            self.rng.shuffle(names)
        for i in range(0, len(names), self.k):
            chunk = names[i:i + self.k]
            batch = Batch.from_data_list([self.s[n] for n in chunk])
            offs, rows, local = batch.ptr[:-1].tolist(), [], []
            for off, n in zip(offs, chunk):
                r = self.by_name[n]
                rows.append(r)
                local.append(self.df.node_idx.values[r] + off)
            yield batch, np.concatenate(rows), torch.from_numpy(np.concatenate(local)).long()


def evaluate(model, head, structures, df, dev, n_proteins=16):
    model.eval(), head.eval()
    preds = np.zeros(len(df), dtype=np.float32)
    with torch.no_grad():
        for batch, rows, local in ProteinBatches(structures, df, n_proteins, shuffle=False):
            h = node_embeddings(model, batch.to(dev))
            p = head(h[local.to(dev)], aa_idx(df.wt_aa.values[rows]).to(dev),
                     aa_idx(df.mut_aa.values[rows]).to(dev))
            preds[rows] = p.float().cpu().numpy()
    y = df.ddG.values
    per = [spearmanr(y[i], preds[i]).statistic
           for i in (np.flatnonzero((df.WT_name == n).values) for n in df.WT_name.unique())
           if len(i) > 5]
    return dict(
        spearman=float(spearmanr(y, preds).statistic),
        pearson=float(pearsonr(y, preds)[0]),
        rmse=float(np.sqrt(np.mean((y - preds) ** 2))),
        spearman_per_protein=float(np.nanmean(per)),
    ), preds


def train_ddg(model, structures, train_df, val_df, cfg, epochs=15, lr=1e-3, freeze=False,
              n_proteins=8, verbose=True, seed=0):
    """Fine-tune (or linear-probe) the pretrained encoder on ddG. Returns (head, history)."""
    torch.manual_seed(seed)
    dev = cfg.device if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    head = DDGHead(cfg.dim).to(dev)

    model.patch_gnn.requires_grad_(not freeze)
    params = list(head.parameters()) + ([] if freeze else list(model.patch_gnn.parameters()))
    opt = torch.optim.Adam(params, lr=lr)
    loader = ProteinBatches(structures, train_df, n_proteins, seed=seed)

    hist, best = [], None
    for ep in range(epochs):
        model.train() if not freeze else model.eval()
        head.train()
        tot = n = 0
        for batch, rows, local in loader:
            batch = batch.to(dev)
            with torch.set_grad_enabled(not freeze):
                h = node_embeddings(model, batch)
            if freeze:
                h = h.detach()
            pred = head(h[local.to(dev)], aa_idx(train_df.wt_aa.values[rows]).to(dev),
                        aa_idx(train_df.mut_aa.values[rows]).to(dev))
            y = torch.from_numpy(train_df.ddG.values[rows]).float().to(dev)
            loss = nn.functional.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(rows)
            n += len(rows)
        m, _ = evaluate(model, head, structures, val_df, dev)
        m.update(epoch=ep, train_mse=tot / max(n, 1))
        hist.append(m)
        # Select on VALIDATION and restore that checkpoint before touching test. Evaluating
        # the last epoch instead lets per-epoch val noise (~0.05 rho here) leak straight into
        # the reported test number, which is most of the between-arm difference we care about.
        if best is None or m["spearman"] > best["spearman"]:
            best = m
            best_state = ({k: v.detach().clone() for k, v in head.state_dict().items()},
                          {k: v.detach().clone() for k, v in model.patch_gnn.state_dict().items()})
        if verbose:
            logger.info(f"  ep {ep:>3}  mse {m['train_mse']:.4f}  val rho {m['spearman']:.4f}  "
                  f"per-prot {m['spearman_per_protein']:.4f}  rmse {m['rmse']:.4f}")

    head.load_state_dict(best_state[0])
    model.patch_gnn.load_state_dict(best_state[1])
    return head, hist, best
