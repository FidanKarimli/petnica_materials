
my workspace for everything from the Petnica Summer Institute ML for Science track:
course notebooks, lecture slides, project code, and whatever else I pick up along the way.

## What's in here

```
petnica_2026/
├── 02_08_intro_task_temp_prediction/   2026 warm-up task: predict Petnica's next-day temperature
│   ├── README.md                        assignment description
│   ├── task.pdf                          assignment sheet (PDF)
│   └── data/petnica_daily.csv            365 days of ERA5 weather data for Petnica
│
├── 03_08_pinns/                        2026 physics-informed neural network notebooks
│   ├── 01_setting_up_the_problem.ipynb
│   ├── 02_failure_modes_and_repair.ipynb
│   ├── 03_expanding_dimensionality.ipynb
│   ├── 05_pinns_on_curved_surfaces.ipynb
│   └── pinn_damped_harmonic_oscillator.ipynb
│
├── 06_08_bayesian_dl_project/          2026 "Honest Error Bars" Bayesian deep learning project
│   └── RESOURCES/
│       ├── ABSTRACT.md, ASSIGNMENT.md, RUBRIC.md   project spec and grading rubric
│       └── starter/                     provided starter code (uv project)
│           ├── bdl/                     data loaders, models, metrics, plotting (not to edit)
│           └── notebooks/                01_deterministic.ipynb … 08_bonus_gpu.ipynb
│
├── gnn-pde-tutorial/                    Standalone tutorial: GNNs (GCN/GAT) for solving PDEs
│   ├── models.py                        GCN / GAT model definitions
│   ├── PSI_SciML2026_Graph_Neural_Networks_for_solving_PDEs.ipynb
│   └── AllenCahn_NEW.h5                 model weights / data for the Allen-Cahn example
│
├── notes/                               Reference slides and lecture notes (PDFs)
│   ├── bayesian_ml.pdf
│   ├── braonic_slides_35h.pdf
│   ├── PINNs.pdf
│   └── VGligorijevic-PSI-2025_ML-2025 (1).pdf   slides accompanying the protein-folds project
│
├── PSI-ML_2025-protein-folds/           Protein fold classification with graph neural
│                                        networks (regular tracked folder)
│
└── sciml26_gjepa/                       Final project (with teammate Pau Martinez): see below
```


## Final project: Graph-JEPA for protein stability

`sciml26_gjepa/` is our final project for the Petnica Summer Institute Scientific
Machine Learning track (August 2026), built together with my teammate **Pau
Martinez**.

We train a graph neural network in two stages: first it's **pretrained with no
labels at all**, using a Joint-Embedding Predictive Architecture (JEPA) on ~6,800
unlabeled protein structures (SCOP, via ProteinShake) to learn general-purpose
structural representations. We then **fine-tune** that pretrained model on the
MegaScale dataset (862 proteins, 271k stability measurements) to predict **ddG** —
how much a single amino-acid mutation destabilizes a protein — and compare against
ThermoMPNN's published benchmark (Spearman 0.642 from scratch / 0.725 pretrained on
the full PDB) to see how far a small, self-pretrained model can get on a
protein-disjoint split.

Repo: [paumartinez160/sciml26_gjepa](https://github.com/paumartinez160/sciml26_gjepa)

## Where things came from

Most of this is not my original material. I pulled it together from different course repos
and lecture sessions, reorganized it into one place, and added my own notes on top.

| Folder | Source | What I did |
|---|---|---|
| `02_08_intro_task_temp_prediction/`, `03_08_pinns/`, `06_08_bayesian_dl_project/` | [petnica-ml-sci/ml-sci-materials](https://github.com/petnica-ml-sci/ml-sci-materials) | Extracted from the official 2026 course repo |
| `PSI-ML_2025-protein-folds/` | [VGligorijevic/PSI-ML_2025-protein-folds](https://github.com/VGligorijevic/PSI-ML_2025-protein-folds) | Pulled from the 2025 project repo |
| `gnn-pde-tutorial/`, `notes/` | Petnica lecture materials | Collected from course sessions |
| `sciml26_gjepa/` | [paumartinez160/sciml26_gjepa](https://github.com/paumartinez160/sciml26_gjepa) | Our own final project — built by me and Pau Martinez, not pulled from a course repo |

## A few things to know

`PSI-ML_2025-protein-folds/` was originally its own git clone. I removed its `.git`
and re added everything as normal tracked files, so it lives cleanly inside this repo now.

The three 2026 folders (`02_08_intro_task_temp_prediction/`, `03_08_pinns/`,
`06_08_bayesian_dl_project/`) came from the `ml-sci-materials` repo. That original
clone, including the `sciml_2025/` content, is not on disk right now. I still need
to restore it and reconcile it with what's already here.