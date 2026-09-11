"""Protein residue graphs: MegaScale (downstream ddG) and SCOP (pretraining corpus).

Verified facts about MegaScale that the loaders depend on -- see
tests/test_protein_correctness.py, which re-checks them rather than trusting this comment:

  * `mut_type` ("E1Q") is 1-based into the WILD-TYPE structure's residue sequence, which is
    contiguous from resid 1. Checked on ~2400 mutations: 0 mismatches.
  * `aa_seq` is the MUTANT sequence, NOT the wild type. It differs from the structure at
    exactly the mutated position. Using it as the reference is an easy, silent, fatal bug.
  * Splits are protein-disjoint (239/31/28 train/val/test WT proteins, 0 overlap),
    matching the ThermoMPNN homology-aware clustering.
  * Rows are heavily duplicated: 1.29M of 1.5M train rows repeat a (WT_name, mut_type).
"""
import os
import re
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
UNK = len(AA)                                   # 20
N_AA = len(AA) + 1

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

HF_REPO = "RosettaCommons/MegaScale"
MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_pdb(text):
    """CA-only parse -> (sequence, coords[N,3], resnums[N]). Keeps the first altloc."""
    res = {}
    for ln in text.splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            n = int(ln[22:26])
            if n not in res:
                res[n] = (AA3.get(ln[17:20].strip(), "X"),
                          (float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    ks = sorted(res)
    return ("".join(res[k][0] for k in ks),
            np.array([res[k][1] for k in ks], dtype=np.float32),
            np.array(ks, dtype=np.int64))


def seq_to_idx(seq):
    return torch.tensor([AA_IDX.get(c, UNK) for c in seq], dtype=torch.long)


def residue_graph(seq, coords, k=16, rwse_dim=16, max_dist=None):
    """kNN contact graph over CA atoms.

    Computed with a dense cdist rather than torch_cluster.knn_graph: proteins here are
    <=~500 residues so the N^2 matrix is trivial, and it keeps the install dependency-free
    (torch-cluster is deprecated in PyG 2.8).
    """
    from ..patch import random_walk_se

    pos = torch.from_numpy(np.ascontiguousarray(coords)).float()
    n = pos.size(0)
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float("inf"))
    kk = min(k, n - 1)
    if kk < 1:                                            # single-residue edge case
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        idx = d.topk(kk, largest=False).indices            # [N, k]
        dst = torch.arange(n).unsqueeze(1).expand(-1, kk).reshape(-1)
        src = idx.reshape(-1)
        if max_dist is not None:
            keep = d[dst, src] <= max_dist
            src, dst = src[keep], dst[keep]
        edge_index = torch.stack([src, dst])
        # symmetrise + dedupe so message passing is undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], 1)
        edge_index = torch.unique(edge_index, dim=1)

    g = Data(x=seq_to_idx(seq).unsqueeze(1), edge_index=edge_index, pos=pos, num_nodes=n)
    s, t = edge_index
    dist = (pos[s] - pos[t]).norm(dim=-1, keepdim=True) if edge_index.numel() else torch.zeros(0, 1)
    sep = (s - t).abs().float().unsqueeze(-1) if edge_index.numel() else torch.zeros(0, 1)
    g.edge_attr = torch.cat([dist, sep.clamp(max=32.0)], -1)     # continuous, projected later
    g.rwse = random_walk_se(edge_index, n, rwse_dim)
    g.seq = seq
    return g


# --------------------------------------------------------------------- MegaScale
def load_megascale_structures(root="data", k=16, rwse_dim=16):
    """-> {WT_name: Data}. 862 AlphaFold models, ~60MB, one HF call."""
    from datasets import load_dataset

    os.environ.setdefault("HF_HOME", os.path.join(root, "hf"))
    rows = load_dataset(HF_REPO, name="AlphaFold_model_PDBs", data_dir="AlphaFold_model_PDBs")["train"]
    out = {}
    for r in rows:
        seq, coords, _ = parse_pdb(r["pdb"])
        if len(seq) == 0:
            continue
        g = residue_graph(seq, coords, k=k, rwse_dim=rwse_dim)
        g.name = r["name"]
        out[r["name"]] = g
    return out


def load_megascale_ddg(root="data", split=None, dedup=True):
    """-> DataFrame[WT_name, mut_type, pos, wt_aa, mut_aa, ddG, split].

    ddG sign follows the source `ddG_ML`: NEGATIVE = destabilising (77% of the data).
    """
    from datasets import load_dataset

    os.environ.setdefault("HF_HOME", os.path.join(root, "hf"))
    ds = load_dataset(HF_REPO, name="dataset3_single", data_dir="dataset3_single")
    frames = []
    for sp in (["train", "val", "test"] if split is None else [split]):
        d = ds[sp].to_pandas()[["WT_name", "mut_type", "ddG_ML"]].copy()
        d["split"] = sp
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    m = df.mut_type.str.extract(MUT_RE)
    df["wt_aa"], df["pos"], df["mut_aa"] = m[0], pd.to_numeric(m[1], errors="coerce"), m[2]
    df["ddG"] = pd.to_numeric(df.ddG_ML, errors="coerce")
    df = df.dropna(subset=["wt_aa", "pos", "mut_aa", "ddG"])
    df["pos"] = df["pos"].astype(int)
    if dedup:
        df = df.drop_duplicates(["WT_name", "mut_type"]).reset_index(drop=True)
    return df.drop(columns=["ddG_ML"])


def attach_node_index(df, structures):
    """Map each mutation to its 0-based node index, dropping rows that fail verification.

    This is the join that decides whether the whole project is measuring anything. Any row
    where the structure's residue at `pos` disagrees with `wt_aa` is discarded, not patched.
    """
    keep, node = [], []
    for wt, pos, aa in zip(df.WT_name.values, df.pos.values, df.wt_aa.values):
        g = structures.get(wt)
        ok = g is not None and 1 <= pos <= len(g.seq) and g.seq[pos - 1] == aa
        keep.append(ok)
        node.append(pos - 1 if ok else -1)
    out = df.copy()
    out["node_idx"] = node
    return out[np.array(keep)].reset_index(drop=True)


# --------------------------------------------------------------------- SCOP (pretraining)
def load_scop(root="data/scop", max_len=256, min_len=30, k=16, rwse_dim=16, limit=None):
    """SCOP structures as residue graphs. ~10k proteins, ~4MB download.

    Length-filtered: SCOP's median is 174 residues while MegaScale domains are 35-72, and
    unbounded lengths (max 1303) make batches ragged and slow.
    """
    from proteinshake.datasets import SCOPDataset

    cache = os.path.join(root, f"scop_{min_len}_{max_len}_k{k}_rw{rwse_dim}.pt")
    if os.path.exists(cache):
        out = torch.load(cache, weights_only=False)
        return out if limit is None else out[:limit]

    os.makedirs(root, exist_ok=True)
    out = []
    for p in SCOPDataset(root=root).proteins():
        r = p["residue"]
        seq = "".join(r["residue_type"])
        if not (min_len <= len(seq) <= max_len):
            continue
        coords = np.stack([np.asarray(r["x"]), np.asarray(r["y"]), np.asarray(r["z"])], 1).astype(np.float32)
        if len(coords) != len(seq):
            continue
        out.append(residue_graph(seq, coords, k=k, rwse_dim=rwse_dim))
    torch.save(out, cache)
    return out if limit is None else out[:limit]
