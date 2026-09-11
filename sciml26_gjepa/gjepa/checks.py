"""Automated checks for the notebook exercises.

Each `check_*` takes the student's function or class and runs a handful of properties
against it, printing a pass/fail line per property with a hint when something breaks.

The checks deliberately test PROPERTIES, not just output shapes: permutation equivariance,
isolated-node handling, gradient flow, whether the thing actually looks at its neighbours.
Those are where the real bugs live, and a shape-only check would wave them through.

You are welcome to read this file. It is not cheating; knowing what you are being tested on
is half of knowing what to build.
"""
import torch
from loguru import logger

__all__ = [
    "check_degrees", "check_aggregate_sum", "check_aggregate_mean", "check_multiset_puzzle",
    "check_gnn_layer", "check_global_pool", "check_knn_edges", "check_ema",
    "check_region_mask", "check_jepa_loss", "check_encoder",
]

# a small fixed graph used by several checks:  0-1, 1-2, 2-3, plus isolated node 4
EI = torch.tensor([[0, 1, 1, 2, 2, 3],
                   [1, 0, 2, 1, 3, 2]])
N = 5


def _report(title, results):
    ok = sum(r[0] for r in results)
    for passed, desc, hint in results:
        if passed:
            logger.success(f"  PASS   {desc}")
        else:
            logger.error(f"  FAIL   {desc}")
            if hint:
                logger.warning(f"         {hint}")
    if ok == len(results):
        logger.success(f"{title}  {ok}/{len(results)} checks passed")
    else:
        logger.warning(f"{title}  {ok}/{len(results)} passed, keep going")
    return ok == len(results)


def _case(desc, fn, hint=""):
    """Run one property. `fn` should raise AssertionError (or anything) on failure."""
    try:
        fn()
        return (True, desc, "")
    except AssertionError as e:
        return (False, desc, str(e) or hint)
    except NotImplementedError:
        return (False, desc, "not implemented yet")
    except Exception as e:
        return (False, desc, f"{type(e).__name__}: {e}" + (f"  |  {hint}" if hint else ""))


def _perm_graph(num_nodes, edge_index, seed=0):
    """A relabelling of the same graph. perm[i] is the new name of old node i."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=g)
    return perm, perm[edge_index]


# ---------------------------------------------------------------- Part 1
def check_degrees(fn):
    """fn(edge_index, num_nodes) -> LongTensor[num_nodes] of in-degrees."""
    def shape():
        d = fn(EI, N)
        assert d.shape == (N,), f"expected shape ({N},), got {tuple(d.shape)}"

    def correct():
        d = fn(EI, N).float()
        want = torch.tensor([1., 2., 2., 1., 0.])
        assert torch.allclose(d, want), f"expected {want.tolist()}, got {d.tolist()}"

    def isolated():
        d = fn(EI, N).float()
        assert d[4] == 0, "node 4 has no edges, its degree must be 0 (and not NaN)"

    def permutation():
        perm, ei2 = _perm_graph(N, EI)
        d1, d2 = fn(EI, N).float(), fn(ei2, N).float()
        assert torch.allclose(d1, d2[perm]), (
            "relabelling the nodes should permute the degrees, not change them. "
            "Your function probably depends on node order somewhere.")

    return _report("Exercise 1.1 (degrees)", [
        _case("returns one number per node", shape),
        _case("degrees are correct on the test graph", correct),
        _case("isolated node gets degree 0", isolated),
        _case("permuting the graph permutes the output", permutation),
    ])


# ---------------------------------------------------------------- Part 2
# A DIRECTED graph, used only to catch src/dst confusion. The undirected EI above cannot:
# if every edge has its reverse, aggregating x[src]->dst and x[dst]->src give the same answer,
# so a backwards implementation sails through.
EI_DIRECTED = torch.tensor([[0, 0, 1],
                            [1, 2, 3]])


def _agg_cases(fn, reduce):
    from torch_geometric.utils import scatter

    x = torch.randn(N, 4)

    def direction():
        out = fn(x, EI_DIRECTED, N)
        want = scatter(x[EI_DIRECTED[0]], EI_DIRECTED[1], dim=0, dim_size=N, reduce=reduce)
        assert torch.allclose(out, want, atol=1e-5), (
            "wrong direction. edge_index[0] is the SOURCE and edge_index[1] is the "
            "DESTINATION, so you gather x[edge_index[0]] and scatter into edge_index[1]. "
            "Swapping them looks fine on an undirected graph and is wrong everywhere else.")

    def shape():
        out = fn(x, EI, N)
        assert out.shape == (N, 4), f"expected ({N}, 4), got {tuple(out.shape)}"

    def matches_pyg():
        out = fn(x, EI, N)
        want = scatter(x[EI[0]], EI[1], dim=0, dim_size=N, reduce=reduce)
        assert torch.allclose(out, want, atol=1e-5), (
            "does not match torch_geometric.utils.scatter. Check that you aggregate "
            "x[src] INTO dst, and not the other way round.")

    def isolated():
        out = fn(x, EI, N)
        assert torch.isfinite(out).all(), (
            "node 4 has no neighbours. Sum should give 0; mean must not divide by zero "
            "and produce NaN. Clamp the denominator to a minimum of 1.")
        assert torch.allclose(out[4], torch.zeros(4)), "isolated node should aggregate to 0"

    def permutation():
        perm, ei2 = _perm_graph(N, EI)
        out1 = fn(x, EI, N)
        out2 = fn(x_permuted(x, perm), ei2, N)
        assert torch.allclose(out1, out2[perm], atol=1e-5), (
            "relabelling nodes should permute the output rows, nothing else")

    return [shape, matches_pyg, direction, isolated, permutation]


def x_permuted(x, perm):
    """Row i of x moves to row perm[i]."""
    out = torch.empty_like(x)
    out[perm] = x
    return out


def check_aggregate_sum(fn):
    """fn(x, edge_index, num_nodes) -> sum of neighbour features per node."""
    shape, matches, direction, isolated, permutation = _agg_cases(fn, "sum")
    return _report("Exercise 2.1 (sum aggregation)", [
        _case("output has one row per node", shape),
        _case("matches PyTorch Geometric", matches),
        _case("aggregates in the right direction", direction),
        _case("isolated node aggregates to zero", isolated),
        _case("permutation equivariant", permutation),
    ])


def check_aggregate_mean(fn):
    """fn(x, edge_index, num_nodes) -> mean of neighbour features per node."""
    shape, matches, direction, isolated, permutation = _agg_cases(fn, "mean")
    return _report("Exercise 2.2 (mean aggregation)", [
        _case("output has one row per node", shape),
        _case("matches PyTorch Geometric", matches),
        _case("aggregates in the right direction", direction),
        _case("isolated node gives 0, not NaN", isolated),
        _case("permutation equivariant", permutation),
    ])


def check_multiset_puzzle(mean_blind, max_blind):
    """mean_blind = (X, Y) that MEAN cannot tell apart but MAX can.
       max_blind  = (X, Y) that MAX cannot tell apart but MEAN can."""
    def unpack(pair, name):
        X, Y = pair
        X, Y = torch.as_tensor(X, dtype=torch.float), torch.as_tensor(Y, dtype=torch.float)
        assert X.numel() and Y.numel(), f"{name}: both multisets need at least one element"
        return X, Y

    def a():
        X, Y = unpack(mean_blind, "mean_blind")
        assert torch.allclose(X.mean(), Y.mean()), (
            f"mean({X.tolist()})={X.mean():.3f} but mean({Y.tolist()})={Y.mean():.3f}. "
            "You need two multisets with the SAME mean.")
        assert not torch.allclose(X.max(), Y.max()), (
            "max gives the same answer too, so this pair does not separate them")

    def b():
        X, Y = unpack(max_blind, "max_blind")
        assert torch.allclose(X.max(), Y.max()), (
            f"max({X.tolist()})={X.max():.3f} but max({Y.tolist()})={Y.max():.3f}. "
            "You need two multisets with the SAME max.")
        assert not torch.allclose(X.mean(), Y.mean()), (
            "mean gives the same answer too, so this pair does not separate them")

    return _report("Exercise 2.3 (multiset puzzle)", [
        _case("found a pair mean is blind to but max is not", a),
        _case("found a pair max is blind to but mean is not", b),
    ])


def check_gnn_layer(layer):
    """layer(x, edge_index) -> [num_nodes, out_dim]. Pass an INSTANCE, already built."""
    x = torch.randn(N, 8)

    def runs():
        out = layer(x, EI)
        assert out.dim() == 2 and out.shape[0] == N, (
            f"expected [{N}, out_dim], got {tuple(out.shape)}")

    def uses_neighbours():
        out1 = layer(x, EI)
        x2 = x.clone()
        x2[0] += 10.0                                   # node 0 is a neighbour of node 1
        out2 = layer(x2, EI)
        assert not torch.allclose(out1[1], out2[1], atol=1e-6), (
            "changing node 0 did not change node 1, but they are connected. "
            "Your layer is ignoring the graph.")

    def respects_structure():
        out1 = layer(x, EI)
        x2 = x.clone()
        x2[4] += 10.0                                   # node 4 is isolated
        out2 = layer(x2, EI)
        assert torch.allclose(out1[0], out2[0], atol=1e-6), (
            "changing the isolated node 4 changed node 0. Messages are leaking between "
            "nodes that are not connected.")

    def permutation():
        perm, ei2 = _perm_graph(N, EI)
        out1 = layer(x, EI)
        out2 = layer(x_permuted(x, perm), ei2)
        assert torch.allclose(out1, out2[perm], atol=1e-4), (
            "not permutation equivariant. A GNN layer must not care what order the nodes "
            "are stored in.")

    with torch.no_grad():
        return _report("Exercise 2.4 (a GNN layer)", [
            _case("runs and returns one row per node", runs),
            _case("a node's output depends on its neighbours", uses_neighbours),
            _case("disconnected nodes do not influence each other", respects_structure),
            _case("permutation equivariant", permutation),
        ])


def check_global_pool(fn):
    """fn(x, batch, num_graphs) -> [num_graphs, F], summing node features per graph."""
    x = torch.randn(6, 3)
    batch = torch.tensor([0, 0, 0, 1, 1, 2])

    def shape():
        out = fn(x, batch, 3)
        assert out.shape == (3, 3), f"expected (3, 3), got {tuple(out.shape)}"

    def correct():
        out = fn(x, batch, 3)
        want = torch.stack([x[:3].sum(0), x[3:5].sum(0), x[5:].sum(0)])
        assert torch.allclose(out, want, atol=1e-5), "sums do not match"

    def size_sensitive():
        """Sum pooling must distinguish a graph from two copies of it. Mean cannot."""
        one = fn(torch.ones(2, 3), torch.zeros(2, dtype=torch.long), 1)
        two = fn(torch.ones(4, 3), torch.zeros(4, dtype=torch.long), 1)
        assert not torch.allclose(one, two), (
            "you are averaging, not summing. Sum pooling keeps graph size; mean throws it away.")

    return _report("Exercise 2.5 (readout)", [
        _case("one row per graph", shape),
        _case("sums are correct", correct),
        _case("distinguishes a graph from a bigger one", size_sensitive),
    ])


# ---------------------------------------------------------------- Part 3
def check_knn_edges(fn, k=3):
    """fn(pos, k) -> edge_index [2, E], undirected, no self loops."""
    torch.manual_seed(0)
    pos = torch.randn(20, 3)

    def shape():
        ei = fn(pos, k)
        assert ei.dim() == 2 and ei.shape[0] == 2, f"expected [2, E], got {tuple(ei.shape)}"

    def no_self_loops():
        ei = fn(pos, k)
        assert (ei[0] != ei[1]).all(), "a residue should not be its own neighbour"

    def undirected():
        ei = fn(pos, k)
        e = set(map(tuple, ei.t().tolist()))
        missing = [(a, b) for a, b in e if (b, a) not in e]
        assert not missing, (
            f"{len(missing)} edges have no reverse. Message passing needs both directions, "
            "so add the flipped edge_index and de-duplicate.")

    def nearest():
        ei = fn(pos, k)
        d = torch.cdist(pos, pos)
        node = 0
        nbrs = ei[0][ei[1] == node]
        true_k = d[node].argsort()[1:k + 1]
        assert set(true_k.tolist()) <= set(nbrs.tolist()), (
            f"node 0's {k} nearest neighbours are {true_k.tolist()} but you connected it to "
            f"{sorted(nbrs.tolist())}")

    return _report("Exercise 3.1 (kNN graph)", [
        _case("returns an edge_index", shape),
        _case("no self loops", no_self_loops),
        _case("every edge has its reverse", undirected),
        _case("neighbours really are the nearest ones", nearest),
    ])


# ---------------------------------------------------------------- Part 4
def check_ema(fn):
    """fn(student, teacher, m) updates teacher IN PLACE toward student."""
    def make():
        s = torch.nn.Linear(4, 4)
        t = torch.nn.Linear(4, 4)
        with torch.no_grad():
            for p in s.parameters():
                p.fill_(1.0)
            for p in t.parameters():
                p.fill_(0.0)
        return s, t

    def formula():
        s, t = make()
        fn(s, t, 0.9)
        got = next(t.parameters()).detach()
        assert torch.allclose(got, torch.full_like(got, 0.1), atol=1e-6), (
            f"after one step from 0 toward 1 with m=0.9 the teacher should be 0.1, "
            f"got {got.flatten()[0]:.4f}. The rule is  t = m*t + (1-m)*s.")

    def m_zero_copies():
        s, t = make()
        fn(s, t, 0.0)
        got = next(t.parameters()).detach()
        assert torch.allclose(got, torch.ones_like(got), atol=1e-6), (
            "with m=0 the teacher should become an exact copy of the student")

    def converges():
        s, t = make()
        for _ in range(50):
            fn(s, t, 0.9)
        got = next(t.parameters()).detach()
        assert got.mean() > 0.99, "repeated updates should pull the teacher toward the student"

    def in_place():
        s, t = make()
        before = next(t.parameters())
        fn(s, t, 0.9)
        assert next(t.parameters()) is before, (
            "update the teacher's tensors in place (p.data.mul_/add_), do not rebind them")

    return _report("Exercise 4.1 (EMA update)", [
        _case("uses the right formula", formula),
        _case("m=0 makes teacher a copy of student", m_zero_copies),
        _case("repeated updates converge", converges),
        _case("updates in place", in_place),
    ])


# ---------------------------------------------------------------- Part 5 / 6
def check_region_mask(fn, ratio=0.25):
    """fn(pos, ratio) -> BoolTensor[N], a spatially contiguous region."""
    torch.manual_seed(0)
    pos = torch.randn(200, 3) * 10

    def dtype_and_size():
        m = fn(pos, ratio)
        assert m.dtype == torch.bool, f"return a bool mask, got {m.dtype}"
        assert m.shape == (200,), f"expected shape (200,), got {tuple(m.shape)}"
        frac = m.float().mean().item()
        assert abs(frac - ratio) < 0.05, f"masked {frac:.0%}, expected about {ratio:.0%}"

    def contiguous():
        spreads = []
        for _ in range(10):
            m = fn(pos, ratio)
            spreads.append(torch.cdist(pos[m], pos[m]).mean().item())
        whole = torch.cdist(pos, pos).mean().item()
        assert sum(spreads) / len(spreads) < 0.8 * whole, (
            f"masked residues are spread {sum(spreads)/len(spreads):.1f} apart on average, "
            f"about the same as the whole structure ({whole:.1f}). You are masking at random. "
            "Pick a seed residue and take its nearest neighbours.")

    def varies():
        masks = {tuple(fn(pos, ratio).tolist()) for _ in range(5)}
        assert len(masks) > 1, "the mask is identical every call, it should be random"

    return _report("Exercise 6.1 (region mask)", [
        _case("returns a bool mask of the right size", dtype_and_size),
        _case("masked residues are spatially close together", contiguous),
        _case("a different region each time", varies),
    ])


def check_jepa_loss(fn, encoder, target_encoder, predictor, batch):
    """fn(encoder, target_encoder, predictor, batch, mask) -> scalar loss."""
    mask = torch.zeros(batch.x.size(0), dtype=torch.bool)
    mask[::4] = True

    def scalar():
        loss = fn(encoder, target_encoder, predictor, batch, mask)
        assert loss.dim() == 0, f"loss should be a single number, got shape {tuple(loss.shape)}"
        assert torch.isfinite(loss), "loss is NaN or inf"

    def positive():
        loss = fn(encoder, target_encoder, predictor, batch, mask)
        assert loss.item() > 0, "loss is exactly zero before any training, which is suspicious"

    def no_target_grad():
        for p in target_encoder.parameters():
            p.grad = None
        for p in encoder.parameters():
            p.grad = None
        fn(encoder, target_encoder, predictor, batch, mask).backward()
        leaked = [n for n, p in target_encoder.named_parameters() if p.grad is not None]
        assert not leaked, (
            f"gradient reached the target encoder ({leaked[0]}). Wrap that branch in "
            "torch.no_grad() and/or call .detach(). Without this the model collapses.")

    def encoder_gets_grad():
        got = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in encoder.parameters())
        assert got, "the trainable encoder received no gradient at all"

    def input_untouched():
        before = batch.x.clone()
        fn(encoder, target_encoder, predictor, batch, mask)
        assert torch.equal(batch.x, before), (
            "you modified batch.x in place. Clone it before writing the MASK token, "
            "otherwise the next epoch sees a permanently damaged protein.")

    return _report("Exercise 7.2 (JEPA loss)", [
        _case("returns a finite scalar", scalar),
        _case("loss is non-zero at initialisation", positive),
        _case("no gradient reaches the target encoder", no_target_grad),
        _case("the trainable encoder does get gradient", encoder_gets_grad),
        _case("does not modify the input batch", input_untouched),
    ])


def check_encoder(enc, dim=64):
    """enc(x, edge_index, edge_attr) -> [num_nodes, dim]. Pass a built instance."""
    from .protein.data import load_scop

    g = load_scop(limit=1)[0]

    def shape():
        out = enc(g.x, g.edge_index, g.edge_attr)
        assert out.shape == (g.num_nodes, dim), (
            f"expected ({g.num_nodes}, {dim}), got {tuple(out.shape)}")

    def uses_edges():
        out1 = enc(g.x, g.edge_index, g.edge_attr)
        half = g.edge_index[:, : g.edge_index.size(1) // 2]
        out2 = enc(g.x, half, g.edge_attr[: half.size(1)])
        assert not torch.allclose(out1, out2, atol=1e-5), (
            "removing half the contacts changed nothing, so your encoder is ignoring the graph")

    def uses_edge_attr():
        out1 = enc(g.x, g.edge_index, g.edge_attr)
        out2 = enc(g.x, g.edge_index, g.edge_attr * 3.0)
        assert not torch.allclose(out1, out2, atol=1e-5), (
            "changing the distances changed nothing. Use a conv that consumes edge features "
            "(GINEConv, GATv2Conv, TransformerConv). GCNConv and SAGEConv cannot.")

    def uses_identity():
        x2 = g.x.clone()
        x2[:, 0] = (x2[:, 0] + 5) % 20
        out1 = enc(g.x, g.edge_index, g.edge_attr)
        out2 = enc(x2, g.edge_index, g.edge_attr)
        assert not torch.allclose(out1, out2, atol=1e-5), (
            "changing every amino acid changed nothing, so the residue identity is unused")

    with torch.no_grad():
        return _report("Exercise 7.1 (protein encoder)", [
            _case("one embedding per residue", shape),
            _case("uses the contact graph", uses_edges),
            _case("uses the edge features (distances)", uses_edge_attr),
            _case("uses the amino-acid identity", uses_identity),
        ])
