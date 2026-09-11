import os
import pandas as pd

from torch.utils.data import DataLoader
from dataset import PaddCollator, ProtFoldDataset

import pytorch_lightning as pl

from models import ProtFoldClassifier

# fold classes
fold_ids = ['3.40.50', '2.60.40', '3.20.20', '3.30.70', '2.60.120', '1.10.10']

# number of AA
vocab_size=24

# Directory wih PDB files
pdb_dir = "./dompdb/"
# pdb_dir = None
gnn = True

# epochs
epochs=20

# Directory with model checkpoints
checkpoint_path = os.path.join('./', 'checkpoints')
if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)
print(f'### Storing checkpoints at {checkpoint_path}')

# load data ############################################
print("# Data ##############")
df = pd.read_csv("./data/protein_fold_classification.csv")
df_train = df[df.partition == 'train']
df_valid = df[df.partition == 'val']
print(f'Train/Valid data size: {df_train.shape} / {df_valid.shape}')


# data loaders ###########################################
collate_padd = PaddCollator(vocab_size=vocab_size)

train_loader = DataLoader(
    ProtFoldDataset(df_train, fold_ids=fold_ids, pdb_dir=pdb_dir),
    batch_size=64,
    collate_fn=collate_padd,
    shuffle=True,
    drop_last=True
    )

valid_loader = DataLoader(
    ProtFoldDataset(df_valid, fold_ids=fold_ids, pdb_dir=pdb_dir),
    batch_size=64,
    collate_fn=collate_padd,
    shuffle=True,
    drop_last=False
    )


# trainer & logger
checkpoint = pl.callbacks.ModelCheckpoint(
    dirpath=checkpoint_path,
    filename="protfold-{step:03d}-{val_loss:0.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        verbose=True
    )
# W&B logger
# logger = pl.loggers.WandbLogger(project='ProtFold', save_dir=os.path.join('./'))

# TensorBoard
logger = pl.loggers.TensorBoardLogger(os.path.join('./tensorboard'))
trainer = pl.Trainer(
    accelerator="cpu",
    logger=logger,
    max_epochs=epochs,
    callbacks=[checkpoint]
    )

# model
model = ProtFoldClassifier(vocab_size=vocab_size, num_classes=len(fold_ids), hidden_dim=64, gnn=gnn)

# train ###################################################
print('# Training commences ##########')
trainer.fit(model, train_loader, valid_loader)
print('# Training done ###############')