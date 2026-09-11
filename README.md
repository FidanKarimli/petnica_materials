
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
└── PSI-ML_2025-protein-folds/           Protein fold classification with graph neural
                                         networks (regular tracked folder)
```


## Where things came from

Most of this is not my original material. I pulled it together from different course repos
and lecture sessions, reorganized it into one place, and added my own notes on top.

| Folder | Source | What I did |
|---|---|---|
| `02_08_intro_task_temp_prediction/`, `03_08_pinns/`, `06_08_bayesian_dl_project/` | [petnica-ml-sci/ml-sci-materials](https://github.com/petnica-ml-sci/ml-sci-materials) | Extracted from the official 2026 course repo |
| `PSI-ML_2025-protein-folds/` | [VGligorijevic/PSI-ML_2025-protein-folds](https://github.com/VGligorijevic/PSI-ML_2025-protein-folds) | Pulled from the 2025 project repo |
| `gnn-pde-tutorial/`, `notes/` | Petnica lecture materials | Collected from course sessions |

## A few things to know

`PSI-ML_2025-protein-folds/` was originally its own git clone. I removed its `.git`
and re added everything as normal tracked files, so it lives cleanly inside this repo now.

The three 2026 folders (`02_08_intro_task_temp_prediction/`, `03_08_pinns/`,
`06_08_bayesian_dl_project/`) came from the `ml-sci-materials` repo. That original
clone, including the `sciml_2025/` content, is not on disk right now. I still need
to restore it and reconcile it with what's already here.