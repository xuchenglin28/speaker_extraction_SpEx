#!/usr/bin/env python

"""
Compute SI-SDR as the evaluation metric
"""

import argparse
from tqdm import tqdm
from collections import defaultdict
from utils.sisdr import sisdr
from utils.audio import WaveReader, Reader

class Report(object):
    def __init__(self, spk2gender=None):
        self.s2g = Reader(spk2gender) if spk2gender else None
        self.snr = defaultdict(float)
        self.cnt = defaultdict(int)

    def add(self, key, val):
        gender = "NG"
        if self.s2g:
            gender = self.s2g[key]
        self.snr[gender] += val
        self.cnt[gender] += 1

    def report(self):
        print("SI-SDR(dB) Report: ")
        for gender in self.snr:
            tot_snrs = self.snr[gender]
            num_utts = self.cnt[gender]
            print("{}: {:d}/{:.3f}".format(gender, num_utts,
                                           tot_snrs / num_utts))


def run(args):
    reporter = Report(args.spk2gender)

    sep_reader = WaveReader(args.sep_scp)
    ref_reader = WaveReader(args.ref_scp)
    for key, sep in tqdm(sep_reader):
        ref = ref_reader[key]
        if sep.size != ref.size:
            end = min(sep.size, ref.size)
            sep = sep[:end]
            ref = ref[:end]
        snr = sisdr(sep, ref)
        reporter.add(key, snr)
    reporter.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to compute SI-SDR, as metric of the separation quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--sep_scp",
        type=str,
        help="Separated speech list, egs: spk1.scp")
    parser.add_argument(
        "--ref_scp",
        type=str,
        help="Reference speech list, egs: ref.scp")
    parser.add_argument(
        "--spk2gender",
        type=str,
        default="",
        help="If assigned, report results per gender")
    args = parser.parse_args()
    run(args)
