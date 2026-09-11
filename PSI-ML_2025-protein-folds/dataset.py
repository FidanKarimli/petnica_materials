import os
from re import I
import torch
from torch.utils import data

import pandas as pd
from utils.util import seq2onehot, pdb2cmap


# top most frequent folds
"""
>>> df['fold_id'].value_counts()[:10]
3.40.50     4715
2.60.40     2074
3.20.20     1104
3.30.70     1011
2.60.120     959
1.10.10      700
3.40.190     500
1.25.40      449
2.40.50      402
1.20.120     388
fold_ids = ['3.40.50', '2.60.40', '3.20.20', '3.30.70', '2.60.120', '1.10.10']
"""


class ProtFoldDataset(data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, fold_ids: list, pdb_dir=None):
        super().__init__()
        self.fold_ids = fold_ids
        self.df = dataframe[dataframe['fold_id'].isin(self.fold_ids)]
        self.pdb_dir = pdb_dir

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # Select sample
        row = self.df.iloc[index]
        prot_id, seq, fold_id = row['prot_id'], row['sequence'], row['fold_id']
        y = torch.tensor(self.fold_ids.index(fold_id), dtype=int)
        H = seq2onehot(seq)
        if self.pdb_dir is None:
            return H, y
        else:
            A = pdb2cmap(os.path.join(self.pdb_dir, prot_id))
            return A, H, y


class PaddCollator(object):
    def __init__(self, vocab_size=20):
        self.vocab_size = vocab_size

    def __call__(self, batch):

        if len(batch[0]) != 3:
            lens = [h.size(0) for (h, _)  in batch]
            max_len = max(lens)
            H_batch = torch.zeros((len(batch), max_len, self.vocab_size), dtype=torch.float32)
            Y_batch = torch.zeros(len(batch), dtype=torch.long)
            for i, (l, x) in enumerate(zip(lens, batch)):
                H_batch[i, :l] = x[0]
                Y_batch[i] = x[1]
            return H_batch, Y_batch
        else:
            lens = [a.size(0) for (a, _, _)  in batch]
            max_len = max(lens)
            H_batch = torch.zeros((len(batch), max_len, self.vocab_size), dtype=torch.float32)
            A_batch = torch.zeros((len(batch), max_len, max_len), dtype=torch.float32)
            Y_batch = torch.zeros(len(batch), dtype=torch.long)
            for i, (l, x) in enumerate(zip(lens, batch)):
                A_batch[i, :l, :l] = x[0]
                H_batch[i, :l] = x[1]
                Y_batch[i] = x[2]
            return (A_batch, H_batch), Y_batch