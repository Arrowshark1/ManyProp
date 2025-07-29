import torch

def evaluate(model, args, data_loader):
    loss_fn = args.get_loss_fn()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batched_graphs, targets, fracs, mixture_sizes in data_loader:
            batch = batched_graphs.to(args().device)
            targets = targets.to(args().device)
            preds = model(batch, mixture_sizes, fracs)

            loss = loss_fn(preds, targets)

            total_loss += loss.item()
    
    return total_loss/len(data_loader.dataset)