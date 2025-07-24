from ManyProp.args import Args
from ManyProp.model.train import run_training
from ManyProp.model.predict import predict

def main():
    args = Args()

    if not args().train:
        smis = args().smiles
        desc = args().descriptors
        fracs = args().mol_fracs
        args.load()
        args().smiles = smis
        args().descriptors = desc
        args().mol_fracs = fracs
        pred, var = predict(args=args)
        print(f"prediciton: {pred}, varience: {var}")

    else:
        run_training(args)



if __name__== '__main__':
    main()