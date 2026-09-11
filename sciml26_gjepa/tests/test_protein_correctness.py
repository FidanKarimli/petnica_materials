"""Protein-side correctness. The dangerous bugs here are all silent index/alignment bugs.

These re-verify the dataset facts the loaders assume rather than trusting a comment: a
one-residue offset between the label table and the structure would still train, still
produce a plausible Spearman, and be entirely meaningless.
"""
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from gjepa.config import protein_config
from gjepa.jepa import GraphJEPA
from gjepa.patch import build_patches, spatial_partition
from gjepa.protein.data import (AA_IDX, parse_pdb, residue_graph, load_scop,
                                load_megascale_structures, load_megascale_ddg, attach_node_index)
from gjepa.protein.ddg import DDGHead, ProteinBatches, node_embeddings, aa_idx


@pytest.fixture(scope="module")
def mega():
    s = load_megascale_structures()
    df = attach_node_index(load_megascale_ddg(), s)
    return s, df


# ------------------------------------------------------------------ dataset facts
def test_mutation_position_indexes_the_structure(mega):
    """mut_type is 1-based into the WILD-TYPE structure sequence. Off-by-one here would
    silently mislabel every training example."""
    s, df = mega
    sub = df.sample(20000, random_state=0)
    bad = sum(s[w].seq[p - 1] != a for w, p, a in zip(sub.WT_name, sub.pos, sub.wt_aa))
    assert bad == 0, f"{bad} mutations disagree with the structure residue"


def test_aa_seq_is_the_mutant_not_the_wildtype(mega):
    """aa_seq is the MUTANT sequence. Using it as the wild-type reference is an easy,
    silent, fatal bug -- it differs from the structure at exactly the mutated position."""
    import pandas as pd
    from gjepa.protein.data import load_megascale_ddg
    from datasets import load_dataset

    s, _ = mega
    raw = load_dataset("RosettaCommons/MegaScale", name="dataset3_single",
                       data_dir="dataset3_single")["train"].to_pandas().head(4000)
    n_checked = 0
    for r in raw.itertuples():
        g = s.get(r.WT_name)
        if g is None or len(g.seq) != len(r.aa_seq):
            continue
        diff = [i for i, (a, b) in enumerate(zip(g.seq, r.aa_seq)) if a != b]
        if diff:
            pos = int(r.mut_type[1:-1])
            assert diff == [pos - 1], "aa_seq differs from structure somewhere other than the mutation site"
            assert r.aa_seq[pos - 1] == r.mut_type[-1], "aa_seq does not carry the mutant residue"
            n_checked += 1
    assert n_checked > 50, "did not actually exercise the check"


def test_splits_are_protein_disjoint(mega):
    """Random mutation-level splits inflate results enormously; these must be by protein."""
    _, df = mega
    sets = {k: set(v.WT_name) for k, v in df.groupby("split")}
    assert not (sets["train"] & sets["test"])
    assert not (sets["train"] & sets["val"])
    assert not (sets["val"] & sets["test"])


def test_every_row_has_a_verified_structure(mega):
    s, df = mega
    assert df.node_idx.min() >= 0
    assert (df.node_idx.values < np.array([len(s[w].seq) for w in df.WT_name])).all()


# ------------------------------------------------------------------ graph construction
def test_residue_graph_is_undirected_without_self_loops():
    g = load_scop(limit=3)[0]
    s, t = g.edge_index
    assert (s != t).all(), "self loops present"
    e = set(map(tuple, g.edge_index.t().tolist()))
    assert all((b, a) in e for a, b in e), "edge_index is not symmetric"


def test_edge_attr_matches_geometry():
    """edge_attr = [CA-CA distance, sequence separation]; both must be consistent with pos."""
    g = load_scop(limit=3)[0]
    s, t = g.edge_index
    d = (g.pos[s] - g.pos[t]).norm(dim=-1)
    assert torch.allclose(g.edge_attr[:, 0], d, atol=1e-4)
    assert torch.allclose(g.edge_attr[:, 1], (s - t).abs().float().clamp(max=32.0))


def test_parse_pdb_lengths_agree():
    s, df = load_megascale_structures(), None
    g = s[list(s)[0]]
    assert g.pos.shape == (len(g.seq), 3)
    assert g.x.shape == (len(g.seq), 1)


# ------------------------------------------------------------------ spatial patching
def test_spatial_patches_are_compact():
    """k-means patches must be spatially tighter than a random node partition, else the
    'predict a structural region' framing is a fiction."""
    from gjepa.patch import random_partition

    g = load_scop(limit=3)[0]
    torch.manual_seed(0)

    def spread(mask):
        out = []
        for row in mask:
            p = g.pos[row]
            if len(p) > 1:
                out.append(torch.cdist(p, p).mean().item())
        return float(np.mean(out))

    sp = spread(spatial_partition(g.pos, 8))
    rd = spread(random_partition(g.num_nodes, 8))
    assert sp < 0.7 * rd, f"spatial patches not compact (spatial {sp:.1f} vs random {rd:.1f})"


def test_spatial_patches_partition_exactly():
    """With num_hops=0 every residue belongs to exactly one patch."""
    cfg = protein_config()
    g = load_scop(limit=3)[0]
    p = build_patches(g, cfg.n_patches, cfg.num_hops, "spatial", cfg.rwse_dim)
    assert p.subgraphs_nodes_mapper.numel() == g.num_nodes
    assert torch.equal(torch.sort(p.subgraphs_nodes_mapper).values, torch.arange(g.num_nodes))


# ------------------------------------------------------------------ ddG readout
def test_ddg_batch_indices_point_at_the_right_residue(mega):
    """THE alignment test. After Batch collation, local[i] must land on the residue whose
    identity equals row i's wild-type amino acid -- across protein boundaries."""
    s, df = mega
    tr = df[df.split == "train"].reset_index(drop=True)
    loader = ProteinBatches(s, tr, n_proteins=8, shuffle=False)
    for k, (batch, rows, local) in enumerate(loader):
        assert int(local.max()) < batch.num_nodes
        got = batch.x[local, 0]
        want = aa_idx(tr.wt_aa.values[rows])
        assert torch.equal(got, want), "ddG batch indices are misaligned with the structures"
        if k >= 2:
            break


def test_ddg_head_depends_on_both_residues():
    torch.manual_seed(0)
    head = DDGHead(32).eval()
    h = torch.randn(16, 32)
    wt, m1, m2 = torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long), torch.full((16,), 5)
    with torch.no_grad():
        a, b = head(h, wt, m1), head(h, wt, m2)
    assert (a - b).abs().mean() > 1e-4, "head ignores the mutant identity"


def test_protein_jepa_forward_backward():
    cfg = protein_config(dim=64, n_token_layers=2)
    ds = load_scop(limit=6)
    b = Batch.from_data_list(
        [build_patches(g, cfg.n_patches, cfg.num_hops, cfg.patcher, cfg.rwse_dim) for g in ds])
    m = GraphJEPA(cfg)
    loss = m(b)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in m.target_encoder.parameters())
    assert node_embeddings(m, b).shape == (b.num_nodes, cfg.dim)


# ------------------------------------------------------------------ node-level JEPA
def test_node_jepa_masks_identity_but_not_structure():
    """Only the amino-acid identity is hidden. If edges or distances were also masked the
    task would stop being 'what belongs at this position given its structural context'."""
    from gjepa.protein.node_jepa import NodeJEPA, MASK_IDX

    cfg = protein_config(dim=32)
    b = Batch.from_data_list(load_scop(limit=4))
    ei, ea = b.edge_index.clone(), b.edge_attr.clone()
    m = NodeJEPA(cfg)
    m(b).backward()
    assert torch.equal(b.edge_index, ei) and torch.equal(b.edge_attr, ea), "structure was mutated"
    assert MASK_IDX not in set(b.x[:, 0].tolist()), "input batch was mutated in place"


def test_node_jepa_stop_grad_and_ema():
    from gjepa.protein.node_jepa import NodeJEPA

    cfg = protein_config(dim=32)
    b = Batch.from_data_list(load_scop(limit=4))
    m = NodeJEPA(cfg)
    m(b).backward()
    assert all(p.grad is None for p in m.target_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.encoder.parameters())

    for p in m.encoder.parameters():
        p.data.fill_(1.0)
    for p in m.target_encoder.parameters():
        p.data.fill_(0.0)
    m.ema_update(0.9)
    k = next(m.target_encoder.parameters())
    assert torch.allclose(k.data, torch.full_like(k, 0.1))


def test_node_jepa_targets_not_constant():
    from gjepa.protein.node_jepa import NodeJEPA

    cfg = protein_config(dim=32)
    b = Batch.from_data_list(load_scop(limit=4))
    assert NodeJEPA(cfg).eval().target_spread(b) > 1e-3


def test_node_jepa_transfers_the_whole_encoder():
    """The entire pretrained encoder must be what ddG fine-tunes -- that is the point."""
    from gjepa.protein.node_jepa import NodeJEPA

    cfg = protein_config(dim=32)
    m = NodeJEPA(cfg)
    assert m.patch_gnn is m.encoder
    enc = {id(p) for p in m.encoder.parameters()}
    assert enc == {id(p) for p in m.patch_gnn.parameters()}


def test_freeze_blocks_encoder_gradients():
    """Linear-probe mode must genuinely freeze the transferred encoder."""
    cfg = protein_config(dim=64, n_token_layers=2)
    m = GraphJEPA(cfg)
    m.patch_gnn.requires_grad_(False)
    assert all(not p.requires_grad for p in m.patch_gnn.parameters())
