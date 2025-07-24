import argparse
import torch.nn as nn
import torch.optim as optim
import json
import types

class Args:
    def __init__(self):
        self.args, _ = self.make_parser()
        if len(self.args.mol_fracs)==self.args.num_mols:
            pass
        elif len(self.args.mol_fracs)+1==self.args.num_mols:
            self.args.mol_fracs.append(1-sum(self.args.mol_fracs))
        else:
            print("Error: missing Mol Fractions")
        

    def __call__(self, *args, **kwds):
        return self.args

    def make_parser(self):
        parser = argparse.ArgumentParser(description = 'training args')
        parser.add_argument('--data_path', type=str, default='./data/data.csv', help='path to data')
        parser.add_argument('--mol_features_path', type=str, default='./data/features.csv', help='path to features')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='path to models')
        parser.add_argument('--num_mols', type=int, default=2, help="number of molecules")
        parser.add_argument('--smiles_columns', nargs='+', type=str, default=["mol1", "mol2"], help="list of columns containing molecule smiles strings")
        parser.add_argument('--targets_column', type=str, default='Property', help="column containing target data")
        parser.add_argument('--mol_features_columns', nargs='+', type=str, default='T', help="list of columns containing molecule descriptors")
        parser.add_argument('--mol_frac_columns', nargs='+', type=str, default=['molfrac'], help="list of columns containing proportion of each molecule")
        parser.add_argument('--epochs', type=int, default=30, help='epochs ot run')
        parser.add_argument('--num_folds', type=int, default=3, help="number of models in ensemble")
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--hidden_size', type=int, default=64, help='shape of hidden layers')
        parser.add_argument('--num_layers', type=int, default=2, help='number of hidden layers')
        parser.add_argument('--input_dim', type=int, default=111, help='number of features')
        parser.add_argument('--device', type=str, default='cuda:0', help='physical device for training')
        parser.add_argument('--loss_function', type=str, default='MSE', help='avaliable: MAE, CrossEntropyLoss, BCE')
        parser.add_argument('--optimizer', type=str, default='Adam', help='avaliable: Adam, SGD')
        parser.add_argument('--metrics', nargs='+', type=str, default=["mse", "r2"], help='testing metrics')
        parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
        parser.add_argument('--data_points', type=int, default=None, help="number of datapoints considered per fold")
        parser.add_argument('--train', type=bool, default=False, help='if the model is being trained the inclusion of additional arguments is recommmended. If not, then all additional arguments will be ignored, being replaced with the arguments stored in the checkpoints directory')
        parser.add_argument('--smiles', nargs='+', type=str, default=['COCCOC', "CCCN"], help="list of smiles (not used in training)")
        parser.add_argument('--descriptors', nargs='+', type=int, default=[298], help="list of descriptors (not used in training)")
        parser.add_argument('--mol_fracs', nargs='+', type=float, default=[0.25], help="list of mol fractions (not used in training)")
        parser.add_argument('--log', type=bool, default=False, help="True if the dataset contains the log of the predicted property")
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
        parser.add_argument('--normalize', type=bool, default=True, help='normalizaiton of target values')
        parser.add_argument('--mean', type=float, default=None, help='mean of target dataset. Calculated if not provided')
        parser.add_argument('--std', type=float, default=None, help='standard deviation of target dataset. Calculated if not provided')
        parser.add_argument('--lightningMPNN', type=bool, default=True, help='if true, uses a MPNN as opposed to a GCN')

        return parser.parse_known_args()

    def save(self):
        with open(f'{self.args.checkpoints_dir}/args.json', 'w') as f:
            json.dump(vars(self.args), f, indent=4)
    
    def load(self):
        with open(f'{self.args.checkpoints_dir}/args.json', 'r') as f:
            loaded_args = json.load(f)
            loaded_args = types.SimpleNamespace(**loaded_args)

            self.args.num_folds = loaded_args.num_folds
            self.args.device = loaded_args.device
            self.args.input_dim = loaded_args.input_dim
            self.args.mean = loaded_args.mean
            self.args.std = loaded_args.std
            self.args.normalize = loaded_args.normalize


    def get_loss_fn(self):
        match self.args.loss_function:
            case 'MSE':
                return nn.MSELoss()
            case 'CrossEntropyLoss':
                return nn.CrossEntropyLoss() 
            case 'SmoothL1':
                return nn.SmoothL1Loss()
            case 'L1':
                return nn.L1Loss()

    
    def get_optimizer(self, params):
        lr = self.args.lr
        weight_decay=self.args.weight_decay
        match self.args.optimizer:
            case 'Adam':
                return optim.Adam(params, lr=lr, weight_decay=weight_decay)
            case 'SGD':
                return optim.SGD(params, lr=lr, weight_decay=weight_decay)
            case _:
                return optim.Adam(params, lr=lr, weight_decay=weight_decay)
