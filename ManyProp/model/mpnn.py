import torch
from torch.nn import Linear, Parameter
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv, SAGEConv
from torch_geometric.utils import add_self_loops, degree

class GCN(torch.nn.Module):
    def __init__(self, in_dim, args, out_dim=1):
        super(GCN, self).__init__()
        self.hidden_dim = args().hidden_size
        self.activation = args.get_activation()
        self.args = args
        self.conv1=SAGEConv(in_dim, self.hidden_dim)
        self.conv2=SAGEConv(self.hidden_dim, self.hidden_dim)
        self.fc=torch.nn.Linear(self.hidden_dim, out_dim)
        self.dropout = torch.nn.Dropout(p=args().dropout)

    def forward(self, data, mixture_sizes, fracs):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        #x = F.leaky_relu(x)
        x = self.activation(x)
        for _ in range(self.args().num_layers):
            x = self.conv2(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        #x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        #mixture_embeddings = torch.split(x, tuple(mixture_sizes.tolist()))

        #fracs = torch.tensor(mol_fracs, device=self.args().device)

        fracs = torch.cat([frac for frac in fracs], dim=0)
        fracs = fracs.to(self.args().device)

        mixture_sizes = mixture_sizes.to(self.args().device)

        sizes = mixture_sizes.sum().item()
        shape = fracs.shape[0]

        if self.args().num_mols ==1:
            x = self.fc(x)
            return x 
        
        assert mixture_sizes.sum().item()==fracs.shape[0]

        mixture_ids = torch.repeat_interleave(
            torch.arange(len(mixture_sizes), device=self.args().device), 
            mixture_sizes)
        contribution_sum_per_mixture = torch.zeros(len(mixture_sizes), device=self.args().device).index_add(0, mixture_ids, fracs)
        normalized_fracs = fracs/contribution_sum_per_mixture[mixture_ids]

        weighted_embeddings = x * normalized_fracs.unsqueeze(1)

        mixture_embeddings = torch.zeros(len(mixture_sizes), x.size(1), device=self.args().device)
        mixture_embeddings = mixture_embeddings.index_add(0, mixture_ids, weighted_embeddings)

        #mixture_embeddings = torch.sum(x*torch.tensor(self.args().mol_fracs).unsqueeze(1), dim=0)

        #mixture_vectors = torch.stack([emb.mean(dim=0) for emb in mixture_embeddings])

        #out = self.fc(mixture_vectors)
        out = self.fc(mixture_embeddings)

        return out
        
class MPNNLayer(MessagePassing):
    def __init__(self, args, in_dim, out_dim):
        super().__init__(aggr='add')
        self.args = args()
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        x=self.lin(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j
    
    def update(self, aggr_out):
        return F.leaky_relu(aggr_out)

class MPNNModel(pl.LightningModule):
    def __init__(self, args, out_dim=1):
        super().__init__()
        self.loss_fn = args.get_loss_fn()
        self.args = args()
        self.dropout = torch.nn.Dropout(p=args().dropout)
        self.mp1 = MPNNLayer(args, in_dim=self.args.input_dim, out_dim=self.args.hidden_size)
        self.mp2 = MPNNLayer(args, in_dim=self.args.hidden_size, out_dim=self.args.hidden_size)
        self.fc = torch.nn.Linear(self.args.hidden_size, out_dim)
        self.training_loss = []
        self.val_loss = []
       
    def forward(self, data, mixture_sizes, fracs):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.mp1(x, edge_index)
        x = F.tanh(x)
        for _ in range(self.args.num_layers):
            x = self.mp2(x, edge_index)
            x = F.tanh(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)

        if self.args.num_mols ==1:
            x = self.fc(x)
            return x

        fracs = torch.cat([frac for frac in fracs], dim=0)
        fracs = fracs.to(self.args.device)

        mixture_sizes = mixture_sizes.to(self.args.device)

        assert mixture_sizes.sum().item()==fracs.shape[0]

        mixture_ids = torch.repeat_interleave(
            torch.arange(len(mixture_sizes), device=self.args.device), 
            mixture_sizes)
        contribution_sum_per_mixture = torch.zeros(len(mixture_sizes), device=self.args.device).index_add(0, mixture_ids, fracs)
        normalized_fracs = fracs/contribution_sum_per_mixture[mixture_ids]

        weighted_embeddings = x * normalized_fracs.unsqueeze(1)

        mixture_embeddings = torch.zeros(len(mixture_sizes), x.size(1), device=self.args.device)
        mixture_embeddings = mixture_embeddings.index_add(0, mixture_ids, weighted_embeddings)

        #mixture_embeddings = torch.sum(x*torch.tensor(self.args().mol_fracs).unsqueeze(1), dim=0)

        #mixture_vectors = torch.stack([emb.mean(dim=0) for emb in mixture_embeddings])

        #out = self.fc(mixture_vectors)
        out = self.fc(mixture_embeddings)

        return out

    def training_step(self, batch, batch_idx):
        batched_graphs, targets, fracs, mixture_sizes = batch
        batched_graphs  = batched_graphs.to(self.args.device)
        targets = targets.view(-1, 1).to(self.args.device)
        outputs = self(batched_graphs, mixture_sizes, fracs)
        loss = self.loss_fn(outputs, targets)
        self.log('train_loss', loss)
        self.training_loss.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batched_graphs, targets, fracs, mixture_sizes = batch
        batched_graphs  = batched_graphs.to(self.args.device)
        targets = targets.view(-1, 1).to(self.args.device)
        outputs = self(batched_graphs, mixture_sizes, fracs)
        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss)
        self.val_loss.append(loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def test_step(self, batch, batch_idx):
        batched_graphs, targets, fracs, mixture_sizes = batch
        preds = self(batched_graphs, mixture_sizes, fracs)
        loss = F.mse_loss(preds, targets)
        self.log("test_loss", loss)

        return loss

    def predict_step(self, batch, batch_idx):
        batched_graphs, targets, fracs, mixture_sizes = batch
        preds = self(batched_graphs, mixture_sizes, fracs)
        return preds

