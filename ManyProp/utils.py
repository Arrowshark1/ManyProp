import random
from ManyProp.data.representation import MoleculeModel
from ManyProp.data.data_pipeline import MixtureDataset
from ManyProp.data.data_pipeline import make_dl




def make_splits(args, data_list):
    random.seed(1231231)

    random.shuffle(data_list)
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    n=len(data_list)
    train_end = int(n*train_ratio)
    val_end = train_end+int(n*val_ratio)

    train_data = data_list[:train_end]
    val_data = data_list[train_end:val_end]
    test_data = data_list[val_end:]

    return train_data, val_data, test_data

def smi_to_ds(smis, tgt, frac, desc):
    data_list = []
    mols = [MoleculeModel(smi, descriptors=desc)() for smi in smis]

    data_list.append((MixtureDataset(mols, float(tgt), frac)))
    data_list = [ds.to_tuple() for ds in data_list]
    return data_list

def single_dl(args, smis, tgt, frac, desc):
    data_list = smi_to_ds(smis=smis, tgt=tgt, frac=frac, desc=desc)
    return make_dl(args=args, data_list=data_list)

