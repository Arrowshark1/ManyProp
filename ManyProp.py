from ManyProp.args import Args
from ManyProp.model.train import run_training, lightning_train
from ManyProp.model.predict import predict

def main():
    args = Args()

    if not args().train:
        args.load()
        pred, var = predict(args=args)
        print(f"prediciton: {pred}, varience: {var}")

    else:
        losses = lightning_train(args) if args().lightningMPNN else run_training(args)



if __name__== '__main__':
    main()