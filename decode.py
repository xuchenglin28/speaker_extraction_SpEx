#!/usr/bin/env python

import os
import argparse

import torch as th
import numpy as np

from nnet.spex_plus import SpEx_Plus
from utils.logger import get_logger
from utils.audio import WaveReader, write_wav
logger = get_logger(__name__)

class NnetComputer(object):
    def __init__(self, cpt_dir, gpuid, nnet_conf):
        self.device = th.device(
            "cuda:{}".format(gpuid)) if gpuid >= 0 else th.device("cpu")
        nnet = self._load_nnet(cpt_dir, nnet_conf)
        self.nnet = nnet.to(self.device) if gpuid >= 0 else nnet
        # set eval model
        self.nnet.eval()

    def _load_nnet(self, cpt_dir, nnet_conf):
        nnet = SpEx_Plus(**nnet_conf)
        cpt_fname = os.path.join(cpt_dir, "best.pt.tar")
        cpt = th.load(cpt_fname, map_location="cpu")

        #model_dict = nnet.state_dict()
        #se_dict = {k: v for k, v in cpt["model_state_dict"].items() if k in model_dict}
        #model_dict.update(se_dict)
        #nnet.load_state_dict(model_dict)

        #cpt = {
        #    "epoch": cpt["epoch"],
        #    "model_state_dict": nnet.state_dict(),
        #    "optim_state_dict": cpt["optim_state_dict"]
        #}
        #th.save(cpt, os.path.join(cpt_dir, "tmp.pt.tar"))

        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            cpt_fname, cpt["epoch"]))
        return nnet

    def compute(self, samps, aux_samps, aux_samps_len):
        with th.no_grad():
            raw = th.tensor(samps, dtype=th.float32, device=self.device)
            aux = th.tensor(aux_samps, dtype=th.float32, device=self.device)
            aux_len = th.tensor(aux_samps_len, dtype=th.float32, device=self.device)
            aux = aux.unsqueeze(0)
            sps,sps2,sps3,spk_pred = self.nnet(raw, aux, aux_len)
            sp_samps = np.squeeze(sps.detach().cpu().numpy())
            return sp_samps


def run(args):
    mix_input = WaveReader(args.input, sample_rate=args.sample_rate)
    aux_input = WaveReader(args.input_aux, sample_rate=args.sample_rate)
    nnet_conf = {
        "L1": int(args.L1 * args.sample_rate),
        "L2": int(args.L2 * args.sample_rate),
        "L3": int(args.L3 * args.sample_rate),
        "N": args.N,
        "B": args.B,
        "O": args.O,
        "P": args.P,
        "Q": args.Q,
        "num_spks": args.num_spks,
        "spk_embed_dim": args.spk_embed_dim,
        "causal": args.causal}
    computer = NnetComputer(args.checkpoint, args.gpu, nnet_conf)
    for key, mix_samps in mix_input:
        aux_samps = aux_input[key]
        logger.info("Compute on utterance {}...".format(key))
        samps = computer.compute(mix_samps, aux_samps, len(aux_samps))
        norm = np.linalg.norm(mix_samps, np.inf)
        samps = samps[:mix_samps.size]
        # norm
        samps = samps * norm / np.max(np.abs(samps))
        write_wav(os.path.join(args.output_dir, "{}.wav".format(key)), samps, sample_rate=args.sample_rate)
    logger.info("Compute over {:d} utterances".format(len(mix_input)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do speech separation in time domain using ConvTasNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, help="Directory of checkpoint")
    parser.add_argument(
        "--input", type=str, required=True, help="Script for input waveform")
    parser.add_argument(
        "--input_aux", type=str, required=True, help="Script for input reference waveform")
    parser.add_argument(
        "--gpu", type=int, default=-1,
        help="GPU device to offload model to, -1 means running on CPU")
    parser.add_argument(
        "--sample_rate", type=int, default=8000, help="Sample rate for mixture input")
    parser.add_argument(
        "--output_dir", type=str, default="spex",
        help="Directory to dump separated results out")
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
    args = parser.parse_args()
    run(args)
