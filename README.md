# petnica_2026

Personal workspace for Petnica Summer Institute ML-for-Science materials: course
content, lecture notes, and standalone project code.

## Structure

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
│   └── PINNs.pdf
│
└── PSI-ML_2025-protein-folds/           Standalone project (own git repo): protein fold
                                         classification with graph neural networks
```

## Notes

- `PSI-ML_2025-protein-folds/` is its own git repository (cloned from
  `VGligorijevic/PSI-ML_2025-protein-folds`) and is kept untouched at the top level
  rather than restructured internally.
- `02_08_intro_task_temp_prediction/`, `03_08_pinns/`, and `06_08_bayesian_dl_project/`
  are 2026 course material that originally lived inside a separate `ml-sci-materials/`
  clone (`github.com/petnica-ml-sci/ml-sci-materials`). That clone — including its
  `sciml_2025/` content — is currently **not present on disk** and still needs to be
  restored (e.g. via `git clone`) and reconciled with these three folders.
