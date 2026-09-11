import torch
import numpy as np
from utils.preprocess import load_pdb

def seq2onehot(seq):
    """Create 24-dim embedding"""
    aa_chars = ['P', 'O', 'Y', 'U', 'I', 'K', 'V', 'Q', 'L', 'X', 'E', 'G', 'D', 'N', 'H', 'M', 'W', 'R', 'C', 'T', 'F', 'B', 'S', 'A']
    vocab_size = len(aa_chars)
    vocab_embed = dict(zip(aa_chars, range(vocab_size)))

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), dtype=int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed[v] for v in seq]
    seq_x = np.array([vocab_one_hot[j, :] for j in embed_x], dtype=float)
    seq_x = torch.tensor(seq_x)
    return seq_x

def coords_to_adjmat(coords, threshold=10.0):
    """Convert 3d coordinates to thresholded adjacency matrix/contact map."""
    A = (torch.cdist(coords, coords, p=2) <= threshold).float()
    return A

def pdb2cmap(pdb_fname, threshold=10.0):
     # load coords from a pdb file
     coords = torch.tensor(load_pdb(pdb_fname)['coords'])

     # compute contact map
     A = coords_to_adjmat(coords, threshold=threshold)

     return A