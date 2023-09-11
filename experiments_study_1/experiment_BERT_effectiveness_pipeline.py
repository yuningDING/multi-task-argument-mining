import argparse
import pandas as pd
from torch.utils.data import DataLoader

from experiment_BERT_effectiveness_util import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=False, default=1e-5)
    parser.add_argument("--input_dir", type=str, default="../feedback-prize-effectiveness/train", required=False)
    parser.add_argument("--max_len", type=int, default=512, required=False)
    parser.add_argument("--train_batch_size", type=int, default=8, required=False)
    parser.add_argument("--val_batch_size", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=1, required=False)
    parser.add_argument("--extra_feature", type=str, default="", required=False)
    parser.add_argument("--plus_text",type=bool,default=False, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load Data
    input_dir = args.input_dir
    train = pd.read_csv("../train.csv")
    validation = pd.read_csv("../validation.csv")
    test = pd.read_csv("../test.csv")

    # Get tokenizer and model
    extra_feature = False
    extra_dim = 0
    if args.extra_feature == "prompt":
        extra_dim = 15
        extra_feature = True
    elif args.extra_feature == "argument":
        extra_dim = 7
        extra_feature = True
    elif args.extra_feature == "both":
        extra_dim = 22
        extra_feature = True

    tokenizer, model = build_model_tokenizer(with_custom_feature=extra_feature, num_extra_dims=extra_dim)

    # Convert Data
    train_dataset = FeedbackPrizeDataset(train,
                                         max_len=args.max_len,
                                         tokenizer=tokenizer,
                                         data_path=input_dir,
                                         plus_text=args.plus_text)

    valid_dataset = FeedbackPrizeDataset(validation,
                                         max_len=args.max_len,
                                         tokenizer=tokenizer,
                                         data_path=input_dir,
                                         plus_text=args.plus_text)

    test_dataset = FeedbackPrizeDataset(test,
                                        max_len=args.max_len,
                                        tokenizer=tokenizer,
                                        data_path=input_dir,
                                        plus_text=args.plus_text)

    train_data_loader = DataLoader(train_dataset,
                                   shuffle=True,
                                   batch_size=args.train_batch_size)

    val_data_loader = DataLoader(valid_dataset,
                                 shuffle=False,
                                 batch_size=args.val_batch_size)

    test_data_loader = DataLoader(test_dataset,
                                  shuffle=False,
                                  batch_size=args.val_batch_size)

    # Put model on device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    # Train model
    model = train_model(n_epochs=args.epochs, train_loader=train_data_loader, val_loader=val_data_loader,
                        test_loader=test_data_loader,
                        model=model, lr=args.lr, device=device)
