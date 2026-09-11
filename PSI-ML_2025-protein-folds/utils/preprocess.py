from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from os.path import splitext, basename

import numpy as np
import pandas as pd
import glob


def load_fasta(fasta_fname):
    domain2seq = {}
    with open(fasta_fname) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            name, sequence = record.id, str(record.seq)
            domain_id = name.split('|')[-1].split('/')[0]
            domain2seq[domain_id] = sequence
    return domain2seq


def load_pdb(pdb_fname):
    p = PDBParser(QUIET=True)
    pdb_id = pdb_fname.split('/')[-1]
    structure = p.get_structure(pdb_id, pdb_fname)
    out = {}
    for chain in structure.get_chains():
        residues = [residue for residue in chain if residue.id[0] == " "]
        seq = seq1(''.join([residue.resname for residue in residues]))
        coords_list = []
        for r in residues:
            if 'CA' in r:
                coords_list.append(r['CA'].get_coord())
            elif 'CB' in r:
                coords_list.append(r['CB'].get_coord())
            else:
                coords_list.append((1000.0, 1000.0, 1000.0))
        coords = np.asarray(coords_list)
    out = {'coords': coords, 'seq': seq}
    return out


def load_cath_classes(fname):
    domain2cath = {}
    with open(fname, 'r') as handle:
        for line in handle:
            if line.startswith('#'):
                pass
            else:
                splitted = line.strip().split()
                domain_id = splitted[0]
                cath_id = ".".join(splitted[1:4])
                domain2cath[domain_id] = cath_id
    return domain2cath


def load_cath_names(fname):
    cath2name = {}
    with open(fname, 'r') as handle:
        for line in handle:
            if line.startswith('#'):
                pass
            else:
                splitted = line.strip().split()
                cath_id = splitted[0]
                name = splitted[2]
                cath2name[cath_id] = name
    return cath2name


if __name__ == "__main__":
    domain2seq = load_fasta('cath-dataset-nonredundant-S40.atom.fa')
    domain2cath = load_cath_classes('cath-domain-list.txt')
    cath2name = load_cath_names('cath-names.txt')

    # init dataframe
    df = pd.DataFrame(columns=('prot_id', 'fold_id', 'fold_name', 'seq_len', 'sequence'))

    for pdb_id in glob.glob('./dompdb/*'):
        out = load_pdb(pdb_id)
        domain_id = pdb_id.split('/')[-1]
        seq = out['seq']
        seq_len = len(seq)
        cath_id = domain2cath[domain_id]
        cath_name = cath2name[cath_id]
        df_row = pd.DataFrame([
                        {
                            'prot_id': domain_id,
                            'fold_id': cath_id,
                            'fold_name': cath_name,
                            'seq_len': seq_len,
                            'sequence': seq
                            }
                        ])
        df = pd.concat([df, df_row])
    df.to_csv('../data/protein_fold_classification.csv', index=False)
