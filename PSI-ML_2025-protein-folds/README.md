# PSI-ML_2025-protein-folds

*PSI-ML_2025-protein-folds* is a pedagogical resource for the attendees of the Machine Learning for Scientific Research summer school (held in Petnica Science Center in 2025). It is meant for beginners and experts to explore the protein fold space with Graph Neural Networks.

## Installation

Clone the repo and install the conda environment:

```bash
$ git clone https://github.com/VGligorijevic/PSI-ML_2025-protein-folds.git
$ cd PSI-ML_2025-protein-folds
$ conda env create -f env.yml
$ conda activate protfold
```

## Data
To download PDB files of the protein domains from the Protein Structure Classification Databese (CATH), run:

```bash
$ wget ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz
$ tar -xvzf cath-dataset-nonredundant-S40.pdb.tgz 
```

## Experiments
- `./data/` contains a pre-processed `.csv` file with a non-redundant set (40% sequence identity) of protein domain IDs and CATH classes (see `./utils/preprocess.py` script)
- `train.py` scripts contains the basic code for setting up a data loader and training a simple ConvNN (use `gnn=False` option and set `pdb_dir=None`) or GraphNN model (use `gnn=True` option and set `pdb_dir=./dompdb/`). The model is trained on top 6 most frequent folds in the dataset. 
-  to viz the loss/accuracy curves and monitor the training of the models use tensorboard: `tensorboard --logdir=tensorboard` (open `http://localhost:6006/` in your browser)  

## Contact
Authors: Vladimir Gligorijevic (vgligorijevic@gmail.com).
We welcome your questions and feedback via email or GitHub Issues.