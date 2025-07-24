import torch
from ManyProp.model.mpnn import GCN
from ManyProp.model.mpnn import MPNNModel
from ManyProp.model.evaluate import evaluate
from ManyProp.model.test import test
from ManyProp.utils import make_splits
from ManyProp.data.data_pipeline import parse_data, make_dl

def train(model, args, data_loader):
    loss_fn = args.get_loss_fn()
    optimizer = args.get_optimizer(params = model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.zero_grad()

    model.train()

    total_loss = 0

    for batched_graphs, targets, fracs, mixture_sizes in data_loader:
        batched_graphs = batched_graphs.to(torch.device(args().device))
        targets = targets.view(-1, 1)
        targets = targets.to(torch.device(args().device))

        outputs = model(batched_graphs, mixture_sizes, fracs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        optimizer.step()

        total_loss+=loss.item()

    return total_loss/len(data_loader.dataset)

def run_training(args):
    losses = []
    for fold in range(args().num_folds):
        model = GCN(in_dim=args().input_dim, args=args).to(torch.device(args().device))

        data_list = parse_data(args=args, num_dp=args().data_points)
        args.save()
        train_data, val_data, test_data = make_splits(args=args, data_list=data_list)
        train_dl = make_dl(args=args, data_list=train_data)
        val_dl = make_dl(args=args, data_list=val_data)
        test_dl = make_dl(args=args, data_list=test_data)


        print(f"fold: {fold}")
        min_val_loss=float('inf')

        best_mod = None

        for epoch in range(args().epochs):
            train_loss = train(model=model, args=args, data_loader=train_dl)
            val_loss = evaluate(model=model, args=args, data_loader=val_dl)

            print(f"epoch: {epoch} train_loss: {train_loss} — val_loss: {val_loss}")

            if val_loss<min_val_loss:
                best_mod = model
                torch.save(model.state_dict(), f"checkpoints/model{fold}.pth")

            min_val_loss=min(val_loss, min_val_loss)

            losses.append([fold, epoch, train_loss, val_loss])
        
        results, test_loss = test(best_mod, args, test_dl)
        print(f"results: {results}, test_loss: {test_loss}")

    return losses

def lightning_train(args):
    pass