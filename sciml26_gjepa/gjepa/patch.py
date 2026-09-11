"""Molecule -> subgraph patches, in the batched form the patch GNN consumes.

The patch GNN must run ONCE over one expanded graph, not once per patch. We therefore
precompute a node-replication map: every (patch, node) pair becomes a row in an expanded
node array, and the induced edges are rewritten into that expanded index space. Looping
over patches instead is ~P times slower.
"""
import torch
from torch_geometric.data import Data


class PatchData(Data):
    """Data subclass teaching PyG how to batch the patch mappers."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "subgraphs_nodes_mapper":
            return self.num_nodes
        if key == "subgraphs_edges_mapper":
            return self.edge_index.size(1)
        if key == "combined_subgraphs":
            return self.subgraphs_nodes_mapper.size(0)
        if key == "subgraphs_batch":
            return int(self.n_patches)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "combined_subgraphs":
            return -1
        return super().__cat_dim__(key, value, *args, **kwargs)


def dense_adj(edge_index, num_nodes):
    a = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    if edge_index.numel():
        a[edge_index[0], edge_index[1]] = True
    return a


def random_partition(num_nodes, n_patches, gen=None):
    """Balanced random assignment of every node to exactly one patch -> [P, N] bool."""
    perm = torch.randperm(num_nodes, generator=gen)
    owner = torch.empty(num_nodes, dtype=torch.long)
    owner[perm] = torch.arange(num_nodes) % n_patches
    m = torch.zeros(n_patches, num_nodes, dtype=torch.bool)
    m[owner, torch.arange(num_nodes)] = True
    return m


def spatial_partition(pos, n_patches, iters=10, gen=None):
    """k-means over 3D coordinates -> compact structural regions.

    The natural protein analogue of an image patch: a patch is a contiguous piece of the
    fold, so "predict this region's latent from the rest" is a structurally meaningful task.
    Random node partitions instead scatter each patch across the whole protein.
    """
    n = pos.size(0)
    k = min(n_patches, n)
    c = pos[torch.randperm(n, generator=gen)[:k]].clone()
    a = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        a = torch.cdist(pos, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = pos[m].mean(0)
    mask = torch.zeros(n_patches, n, dtype=torch.bool)
    mask[a, torch.arange(n)] = True
    return mask


def ego_seeds(num_nodes, n_patches, gen=None):
    """One seed node per patch (with replacement when the molecule is smaller than P)."""
    if num_nodes >= n_patches:
        seeds = torch.randperm(num_nodes, generator=gen)[:n_patches]
    else:
        seeds = torch.randint(num_nodes, (n_patches,), generator=gen)
    m = torch.zeros(n_patches, num_nodes, dtype=torch.bool)
    m[torch.arange(n_patches), seeds] = True
    return m


def expand_hops(mask, adj, num_hops):
    """Grow each patch by `num_hops` hops. Patches overlap after expansion -- intended:
    non-overlapping patches lose every inter-patch edge."""
    m = mask.clone()
    for _ in range(num_hops):
        m = m | (m.float() @ adj.float() > 0)
    return m


def random_walk_se(edge_index, num_nodes, k):
    """RWSE: return probabilities [ (D^-1 A)^i _vv ] for i=1..k. Shape [N, k]."""
    a = torch.zeros(num_nodes, num_nodes)
    if edge_index.numel():
        a[edge_index[0], edge_index[1]] = 1.0
    p = a / a.sum(1, keepdim=True).clamp(min=1)
    out, mk = [], torch.eye(num_nodes)
    for _ in range(k):
        mk = mk @ p
        out.append(mk.diagonal().clone())
    return torch.stack(out, dim=1)


def build_patches(data, n_patches, num_hops=1, patcher="random", rwse_dim=16, gen=None):
    """Attach patch mappers (+ RWSE) to a Data object and return it as PatchData."""
    n, e = data.num_nodes, data.edge_index.size(1)
    adj = dense_adj(data.edge_index, n)

    if patcher == "spatial":
        if getattr(data, "pos", None) is None:
            raise ValueError("patcher='spatial' needs data.pos (3D coordinates)")
        base = spatial_partition(data.pos, n_patches, gen=gen)
    elif patcher == "khop":
        base = ego_seeds(n, n_patches, gen)
    else:
        base = random_partition(n, n_patches, gen)
    mask = expand_hops(base, adj, num_hops)                      # [P, N]

    patch_id, node_id = mask.nonzero(as_tuple=True)              # patch-major order
    m = patch_id.numel()

    # position of each (patch, node) pair inside the expanded node array
    pos = torch.full((n_patches, n), -1, dtype=torch.long)
    pos[patch_id, node_id] = torch.arange(m)

    src, dst = data.edge_index
    edge_in = mask[:, src] & mask[:, dst] if e else torch.zeros(n_patches, 0, dtype=torch.bool)
    patch_e, edge_id = edge_in.nonzero(as_tuple=True)
    combined = torch.stack([pos[patch_e, src[edge_id]], pos[patch_e, dst[edge_id]]]) \
        if edge_id.numel() else torch.zeros(2, 0, dtype=torch.long)

    out = PatchData(**{k: v for k, v in data})
    out.subgraphs_nodes_mapper = node_id
    out.subgraphs_edges_mapper = edge_id
    out.combined_subgraphs = combined
    out.subgraphs_batch = patch_id
    out.patch_valid = mask.any(dim=1)                            # empty patches -> masked out
    out.n_patches = n_patches
    out.rwse = random_walk_se(data.edge_index, n, rwse_dim)
    return out
