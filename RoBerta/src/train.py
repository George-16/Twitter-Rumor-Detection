#!/usr/bin/env python
import argparse
import random
import torch
import pickle
from pathlib import Path
import pytorch_lightning as pl
from model import RumourDetection
from callback import DisplayCallback


def add_model_specific_args(parser):
    parser.add_argument("--random_seed", default=42, type=int)

    parser.add_argument("--learning_rate", default=5e-5, type=float)
    # adjust the learning rate to a small number at the beginning and increase it slowly
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    # a regularization to get rid of over-fitting
    parser.add_argument("--weight_decay", default=0.0, type=float)
    # the number of parallel computations
    parser.add_argument("--num_workers", default=8, type=int)

    # model setups
    parser.add_argument("--pre_trained", default='bert-base-uncased')
    parser.add_argument("--max_length", default=60, type=int)
    # dataset batch size
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--dev_batch_size", default=8, type=int)
    parser.add_argument("--test_batch_size", default=8, type=int)

    parser.add_argument("--output_dir", default=None, type=str, required=True)

    return parser


def save_pickle_file(hparams, path):
    # save the hyper-parameters in a binary serializing approach
    with open(path, 'wb') as f:
        return pickle.dump(hparams, f)


def train_model(model, args):
    # initialize random seed in the model with cuda
    pl.seed_everything(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # add checkpoints callback into the menu
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename='{f1_score:.4f}',
        monitor='f1_score',
        mode='max',
        save_top_k=1,
        verbose=False
    )

    # display callback menu
    display_callback = DisplayCallback()
    total_callbacks = [display_callback, checkpoint_callback]

    # add early stop callback into the menu
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='f1_score',
        mode='max',
        patience=3,
        strict=True,
        verbose=True
    )
    total_callbacks.append(early_stopping_callback)

    # set up trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=False,
        callbacks=total_callbacks,
        precision=16
    )
    # train the model
    trainer.fit(model)

    return trainer


def main(args):
    # construct an output folder
    opt_dir = Path(args.output_dir)
    opt_dir.mkdir(exist_ok=True)

    # initialize the model
    model = RumourDetection(args)

    trainer = train_model(model, args)

    # save hyper-parameters
    save_pickle_file(model.hparams, model.save_hparams)

    # test with the optimal model
    trainer.test(verbose=False, ckpt_path='best')

    return model


if __name__ == "__main__":
    # initialize hyper-parameters
    arg_parser = argparse.ArgumentParser()
    # add python lightening hyper-parameters
    arg_parser = pl.Trainer.add_argparse_args(arg_parser)
    # add specific hyper-parameters
    arg_parser = add_model_specific_args(arg_parser)
    args = arg_parser.parse_args()
    main(args)
