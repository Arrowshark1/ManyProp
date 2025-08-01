import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from ManyProp.data.representation import MoleculeModel
#from ManyProp.utils import normalize 

class MixtureDataset(Dataset):
    def __init__(self, graphs, target, fracs):
        self.graphs = Batch.from_data_list(graphs)
        self.target=target
        self.fracs = fracs
    
    def __getitem__(self, idx):
        mixture, target = self.data[idx]
        return mixture, target
    
    def to_tuple(self):
        return self.graphs, self.target, self.fracs, len(self.graphs)

def collate_mixtures(batch):    
    mixtures = []
    targets = []
    fracs = []
    for ds in batch:
        mixtures.append(ds[0])
        targets.append(ds[1])
    
    fracs = torch.cat([item[2] for item in batch], dim=0)

    all_graphs = [g[0] for g in mixtures]

    mixture_sizes = [len(mixture) for mixture in mixtures]

    batched_graphs = Batch.from_data_list(all_graphs)

    return batched_graphs, torch.tensor(targets, dtype=torch.float), fracs, mixture_sizes

def make_dl(args, data_list):
    data_loader = DataLoader(
        dataset = data_list,
        batch_size=args().batch_size,
        shuffle=True,
        collate_fn=collate_mixtures
    )

    return data_loader

def parse_data(args, num_dp=None, offs=0):
    smi_cols = args().smiles_columns
    frac_cols = args().mol_frac_columns
    feats_cols = args().mol_features_columns
    num_mols = args().num_mols

    assert num_mols == len(smi_cols)
    assert len(frac_cols) == len(smi_cols) or len(frac_cols) == (len(smi_cols)-1)
    
    #infer_frac = len(frac_cols) == (len(smi_cols)-1)

    data = pd.read_csv(args().data_path)

    feat_data = pd.read_csv(args().mol_features_path) if args().mol_features_path != args().data_path else None

    all_data = pd.concat([data, feat_data], axis=1) if feat_data is not None else data

    all_data = all_data.dropna()

    if args().shuffle_data:
        all_data = all_data.sample(frac=1)

    args().mean = np.mean(all_data[args().targets_column]) if args().mean is None else args().mean
    args().std = np.std(all_data[args().targets_column]) if args().std is None else args().std
    
    if num_dp:
        all_data = all_data[offs:offs+num_dp]

    data_list = []

    for _, row in all_data.iterrows():
        smis = [row[col] for col in smi_cols]
        fracs = [row[col] for col in frac_cols] if num_mols > 1 else None
        target = float(row[args().targets_column])

        if args().normalize:
            target = normalize(args, target) 

        if num_mols>1:
            if len(fracs)==args().num_mols:
                pass
            if len(fracs)+1==args().num_mols:
                fracs.append(1-sum(fracs))
            else:
                print("Error: missing Mol Fractions")
        mols = []
        for s in range(num_mols):
            mols.append(MoleculeModel(smis[s], [row[col] for col in feats_cols])())

        data_list.append((MixtureDataset(mols, target, fracs if num_mols>1 else [1.0])))

    data_list = [ds.to_tuple()for ds in data_list]
    return data_list

def normalize(args, t):
    #mean = data.mean()
    #std = data.std()

    #return (data-mean)/std, mean, std
    return (t-args().mean)/args().std

def denormalize(args, t):
    #return data*std + mean
    return t*args().std + args().mean