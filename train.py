#!/usr/bin/env python

import os
import pprint
import argparse

from libs.trainer import SiSnrTrainer
from utils.dataset import make_dataloader
from utils.logger import get_logger

from nnet.spex_plus import SpEx_Plus

logger = get_logger(__name__)

def run(args):
    gpuids = tuple(map(int, args.gpus.split(",")))

    L1 = int(args.L1 * args.sample_rate)
    L2 = int(args.L2 * args.sample_rate)
    L3 = int(args.L3 * args.sample_rate)
    nnet = SpEx_Plus(L1=L1, 
                     L2=L2, 
                     L3=L3, 
                     N=args.N, 
                     B=args.B, 
                     O=args.O, 
                     P=args.P, 
                     Q=args.Q, 
                     num_spks=args.num_spks, 
                     spk_embed_dim=args.spk_embed_dim, 
                     causal=args.causal)
    trainer = SiSnrTrainer(nnet,
                           gpuid=gpuids,
                           checkpoint=args.checkpoint,
                           optimizer=args.optimizer,
                           lr=args.lr,
                           weight_decay=1e-5,
                           clip_norm=None,
                           min_lr=1e-8,
                           patience=2,
                           factor=0.5,
                           logging_period=200,
                           no_impr=6)

    tr_mix_scp = os.path.join(args.train_dir, "mix.scp")
    tr_ref_scp = os.path.join(args.train_dir, "ref.scp")
    tr_aux_scp = os.path.join(args.train_dir, "aux.scp")

    dev_mix_scp = os.path.join(args.dev_dir, "mix.scp")
    dev_ref_scp = os.path.join(args.dev_dir, "ref.scp")
    dev_aux_scp = os.path.join(args.dev_dir, "aux.scp")

    chunk_size = args.chunk_size * args.sample_rate
    train_loader = make_dataloader(train=True,
                                   mix_scp=tr_mix_scp,
                                   ref_scp=tr_ref_scp,
                                   aux_scp=tr_aux_scp,
                                   spk_list=args.spk_list,
                                   sample_rate=args.sample_rate,
                                   batch_size=args.batch_size,
                                   chunk_size=chunk_size,
                                   num_workers=args.num_workers)
    dev_loader = make_dataloader(train=False,
                                 mix_scp=dev_mix_scp,
                                 ref_scp=dev_ref_scp,
                                 aux_scp=dev_aux_scp,
                                 spk_list=args.spk_list,
                                 sample_rate=args.sample_rate,
                                 batch_size=args.batch_size,
                                 chunk_size=chunk_size,
                                 num_workers=args.num_workers)

    trainer.run(train_loader, dev_loader, num_epochs=args.epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to start SpEx_Plus training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpus",
                        type=str,
                        default="0,1",
                        help="Training on which GPUs "
                        "(one or more, egs: 0, \"0,1\")")
    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="Number of training epochs")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Directory to dump models")
    parser.add_argument("--batch-size",
                        type=int,
                        default=16,
                        help="Number of utterances in each batch")
    parser.add_argument("--num-workers",
                        type=int,
                        default=4,
                        help="Number of workers used in data loader")
    parser.add_argument("--L1",
                        type=float,
                        default=0.0025,
                        help="Short window length for high temporal resolution, default 2.5ms.")
    parser.add_argument("--L2",
                        type=float,
                        default=0.01,
                        help="Middle window length for middle temporal resolution, default 10ms.")
    parser.add_argument("--L3",
                        type=float,
                        default=0.02,
                        help="Long window length for low temporal resolution, default 20ms.")
    parser.add_argument("--N",
                        type=int,
                        default=256,
                        help="Number of filters of convolution in speech encoder.")
    parser.add_argument("--B",
                        type=int,
                        default=8,
                        help="Number of TCN blocks in each stack.")
    parser.add_argument("--O",
                        type=int,
                        default=256,
                        help="Number of filters of 1x1 convolution.")
    parser.add_argument("--P",
                        type=int,
                        default=512,
                        help="Number of filters of depthwise convolution.")
    parser.add_argument("--Q",
                        type=int,
                        default=3,
                        help="Kernel size of depthwise convolution.")
    parser.add_argument("--num_spks",
                        type=int,
                        default=101,
                        help="Number of speakers within the training data.")
    parser.add_argument("--spk_embed_dim",
                        type=int,
                        default=256,
                        help="Speaker embedding dimension.")
    parser.add_argument("--causal",
                        type=bool,
                        default=False,
                        help="causal for online or non-causal for offline process.")
    parser.add_argument("--train_dir",
                        type=str,
                        default="data/wsj0_2mix/tr",
                        help="Data folder for training data.")
    parser.add_argument("--dev_dir",
                        type=str,
                        default="data/wsj0_2mix/cv",
                        help="Data folder for development data.")
    parser.add_argument("--spk_list",
                        type=str,
                        default="data/wsj0_2mix_extr_tr.spk",
                        help="List of speakers in the training data.")
    parser.add_argument("--sample_rate",
                        type=int,
                        default=8000,
                        help="Sampling rate.")
    parser.add_argument("--chunk_size",
                        type=int,
                        default=4,
                        help="Duration of a segment.")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--optimizer",
                        type=str,
                        default="adam",
                        help="Optimizer type.")
    args = parser.parse_args()
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))

    run(args)
