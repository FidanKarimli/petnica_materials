# Graph-JEPA for protein stability

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zenwor/sciml26_gjepa/blob/main/notebooks/sciml_graph_jepa.ipynb)

Participant project for the **Petnica Summer Institute, Summer School on Scientific Machine
Learning, August 2026**.

You will build a graph neural network, pretrain it on protein structures with **no labels at
all** using a Joint-Embedding Predictive Architecture, then fine-tune it to predict how much
a single amino-acid mutation destabilises a protein.

Everything happens in one notebook:

```
notebooks/sciml_graph_jepa.ipynb
```

---

## Getting started

### Option A: Google Colab (easiest, and what most of you will use)

1. **Click the "Open in Colab" badge above.** That is the whole first step. You do not clone
   anything yourself, and you do not download anything.
2. **Runtime > Change runtime type > T4 GPU**, then Save. Colab starts you on CPU, where
   everything is roughly ten times slower.
3. **Run the first code cell.** It does three things for you:
   * clones this repository into `/content/sciml26_gjepa`
   * changes into it, so `import gjepa` finds the exercise checkers
   * installs the handful of packages Colab is missing
   It takes about a minute. You will see `Colab setup done` when it has worked.
4. Run the rest of the notebook top to bottom.

**Where do the checkers come from?** From that clone. `check_aggregate_sum` and the rest live
in `gjepa/checks.py` inside the repository, which is why step 3 has to happen before anything
else. If you skip it you will get `ModuleNotFoundError: No module named 'gjepa'`.

**If Colab disconnects** (it does, after idling) you lose the whole runtime, including the
clone and the installed packages. Re-run the first cell, then re-run your work. Save your own
edits with File > Save a copy in Drive, or Colab will lose them too.

Colab already has PyTorch installed with a CUDA build matched to its driver, so we
deliberately do not touch torch. Installing our own would break the runtime.

### Option B: your own machine

```bash
git clone https://github.com/zenwor/sciml26_gjepa.git
cd sciml26_gjepa
./setup.sh
```

That creates a virtual environment, installs PyTorch and everything else, and registers a
Jupyter kernel. Then open the notebook and choose **Kernel > Change kernel >
`Python (sciml-gjepa)`**.

No GPU? `TORCH_BACKEND=cpu ./setup.sh` works, it is just slow.

---

## What you will need to know

Nothing beyond basic PyTorch. The notebook introduces graphs, graph neural networks,
self-supervised learning and JEPA from the beginning, with a demo you can break and an
exercise you write for each idea.

There are **11 exercises**, each with an automated checker:

```python
check_aggregate_sum(aggregate_sum)
```

```
SUCCESS |   PASS   output has one row per node
SUCCESS |   PASS   matches PyTorch Geometric
ERROR   |   FAIL   aggregates in the right direction
WARNING |          wrong direction. edge_index[0] is the SOURCE and edge_index[1] is the
                   DESTINATION, so you gather x[edge_index[0]] and scatter into edge_index[1].
```

The checkers live in [`gjepa/checks.py`](gjepa/checks.py). Read them whenever you like.
Knowing what you are being tested on is half of knowing what to build.

---

## The data

Both datasets download themselves on first use. Nothing to set up.

| | what | size |
|---|---|---|
| **MegaScale** (Tsuboyama et al., *Nature* 2023) | 862 protein structures and 271,231 stability measurements | ~60 MB |
| **SCOP** (via ProteinShake) | 6,782 protein structures, no labels, the pretraining corpus | ~4 MB |

The splits are **protein-disjoint**: no protein in the test set appears in training.

## The number you are aiming at

ThermoMPNN (PNAS 2024) used this exact dataset and split:

| | Spearman |
|---|---|
| their model, from scratch | **0.642** |
| their model, with pretrained weights | **0.725** |

You will not beat 0.725, since that model is pretrained on the entire Protein Data Bank. The
question is how far a small model you train yourself gets, and whether pretraining helps.

---

## Layout

```
notebooks/
  sciml_graph_jepa.ipynb    the project, this is the file you work in
  sciml_graph_jepa.py       the same content as plain Python, for clean diffs
  figures/                  diagrams used in the text
gjepa/
  checks.py                 the 11 exercise checkers
  config.py                 every knob, in one dataclass
  encoders.py               swappable GNN backbones and the input encoders
  jepa.py, patch.py         shared JEPA machinery
  protein/
    data.py                 structures, residue graphs, both dataset loaders
    node_jepa.py            the reference JEPA, for comparing against yours
    ddg.py                  the ddG head, fine-tuning and metrics
tests/                      correctness tests for the library
requirements.txt, setup.sh
```

The `.ipynb` and `.py` are kept in sync by [jupytext](https://jupytext.readthedocs.io/).
Edit either. If you edit the `.py` in a text editor, run `jupytext --sync` afterwards.

---

## Two things that will save you a day

**Watch the embedding spread, not the loss.** Self-supervised models fail by collapsing: every
input maps to nearly the same vector, the loss goes to zero, and nothing has been learned. The
loss curve looks *better* when this happens. Demo 4.1 shows it live.

**Keep `num_workers=0` if you are on WSL2.** DataLoader workers forked from a CUDA process
corrupt GPU memory accounting there and produce phantom out-of-memory errors that report
impossible numbers. On Colab you can raise it to 2.

---

## Running the tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Do this when the GPU is idle. Running heavy CPU work next to a training job has killed runs
here before.
