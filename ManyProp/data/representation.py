from rdkit import Chem
from rdkit.Chem import GetPeriodicTable
from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch

class MoleculeModel:
    def __init__(self, smi, descriptors):
        self.mol = Chem.MolFromSmiles(smi)
        #self.frac = frac
        self.descriptors = descriptors
    def __call__(self, *args, **kwds):
        return self.mol_to_data()

    def atom_features(self, atom):
        #return torch.tensor([
        #    atom.GetAtomicNum(),
        #    atom.GetDegree(),
        #    int(atom.GetIsAromatic()),
        #    atom.GetHybridization().real,
        #    atom.GetTotalNumHs(),
        #], dtype=torch.float)


        features = []
        features += one_hot(atom.GetAtomicNum(), list(range(1, 100)))
        features += one_hot(atom.GetDegree(), list(range(1, 5)))
        features += one_hot(atom.GetFormalCharge(), [-1, 0, 1])
        features += [int(atom.GetIsAromatic())]
        features += one_hot(atom.GetHybridization().real, list(range(0,4)))
        return features
        

    
    def bond_features(self, bond):
        bt=bond.GetBondType()
        return torch.tensor([
            bt==Chem.rdchem.BondType.SINGLE,
            bt==Chem.rdchem.BondType.DOUBLE,
            bt==Chem.rdchem.BondType.TRIPLE,
            bt==Chem.rdchem.BondType.AROMATIC
        ], dtype=torch.float)
    
    def mol_to_data(self):
        #x=torch.stack([self.atom_features(atom) for atom in self.mol.GetAtoms()])
        x = torch.tensor([self.atom_features(atom) for atom in self.mol.GetAtoms()], dtype=torch.float)

        edge_index = []
        edge_attr = []

        norm_desc = []

        for bond in self.mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_index += [[i,j], [j,i]]
            bf = self.bond_features(bond)
            edge_attr+=[bf, bf]
        
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr=torch.stack(edge_attr) if edge_attr else torch.zeros((0, 4))

        data = Data(x=x,edge_index=edge_index, edge_attr=edge_attr, graph_features=self.descriptors)

        #data.weight = self.frac

#        print("num atms: ", len(self.mol.GetAtoms()))
#        print("data.x: ", x)
#        print("x.shape: ", x.shape)
#        print("x.sum: ", x.sum())

        return data

def create_batch(smis):
    mol_list = []
    for smi in smis:
        mod = MoleculeModel(smi=smi)
        mol_list.append(mod())
    batch = Batch.from_data_list(mol_list)   
    return batch

def one_hot(x, allowable_set):
    if x not in allowable_set:
        x=allowable_set[-1]
    return [int(x==s) for s in allowable_set]