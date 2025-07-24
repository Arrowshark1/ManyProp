import torch
from ManyProp.data.data_pipeline import denormalize
from ManyProp.model.mpnn import GCN
from ManyProp.utils import single_dl
import os
import numpy as np

def predict(args):
    mods = []
    for m in range(args().num_folds):
        model = GCN(in_dim=args().input_dim, args=args)
        model.to(args().device)

        path = f"{args().checkpoints_dir}/model{m}.pth"
     
        if os.path.exists(path=path):
            state_dict = torch.load(path)
            #model.load_state_dict(state_dict=state_dict)
            print("model_loaded: ", model.load_state_dict(state_dict=state_dict, strict=False))
        else:
            print("failed to parse checkpoints dir")
            return
        mods.append(model)
        
    data_loader = single_dl(args, args().smiles, tgt=0.0, frac=args().mol_fracs, desc=args().descriptors) #target unused

    all_preds = []
    for m in range(args().num_folds):
        model = mods[m]
        model.eval()

        with torch.no_grad():
            for batched_graphs, targets, fracs, mixture_sizes in data_loader:
                batched_graphs = batched_graphs.to(args().device)
                targets = targets.to(args().device)

                pred = model(batched_graphs, mixture_sizes, fracs)

                all_preds.append(pred.item())
    
    prediction = np.mean(all_preds)
    var = np.var(all_preds)

    prediction = prediction.item()

    if args().normalize:
        prediction = denormalize(args, prediction)

    if args().log:
        prediction = 10**prediction

    return prediction, var
            

def predict_graph(args):
    predictions = [] 
    vars = []

    fracs = [x/10.0 for x in range(1,10)]

    for f in fracs:
        args().mol_fracs = [f, 1.0-f]
        pred, var = predict(args=args)
    
        predictions.append(pred)
        vars.append(var.item())

    return fracs, predictions, vars 



