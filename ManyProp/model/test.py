import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def test(model, args, data_loader):
    model.eval()
    loss_fn = args.get_loss_fn()
    test_preds = []
    test_targets = []
    total_loss = 0

    with torch.no_grad():
        for batched_graphs, targets, fracs, mixture_sizes in data_loader:
            batch = batched_graphs.to(args().device)
            targets = targets.to(args().device)
            preds = model(batch, mixture_sizes, fracs)
            
            loss = loss_fn(preds, targets)

            total_loss += loss.item()

            test_preds.append(preds)
            test_targets.append(targets)
        
    avg_loss = total_loss/len(data_loader.dataset)
    test_preds = torch.concat(test_preds, dim=0).cpu()
    test_targets = torch.concat(test_targets, dim=0).cpu()
    results = metrics(args, test_preds, test_targets) 
    return results, avg_loss

        

def metrics(args, preds, targets):
    result = {}
    for metric in args().metrics:
        match metric:
            case 'mse': 
                result['mse'] = mean_squared_error(targets.numpy(), preds.numpy())
            case 'mae':
                result['mae'] = mean_absolute_error(targets.numpy(), preds.numpy())
            case 'r2':
                result['r2'] = r2_score(targets.numpy(), preds.numpy())               
    return result