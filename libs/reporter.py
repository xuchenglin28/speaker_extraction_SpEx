#!/usr/bin/env python

import sys
sys.path.append("../")

from utils.timer import Timer

class Reporter(object):
    """
    A progress reporter to record the loss for each batch
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.loss = []
        self.ce = []
        self.ncorrect = []
        self.nsample = []
        self.timer = Timer()

    def add(self, loss, ce, ncorrect, nsample):
        self.loss.append(loss)
        self.ce.append(ce)
        self.ncorrect.append(ncorrect)
        self.nsample.append(nsample)
        N = len(self.loss)
        if not N % self.period:
            avg = sum(self.loss[-self.period:]) / self.period
            avg_ce = sum(self.ce[-self.period:]) / self.period
            acc = sum(self.ncorrect[-self.period:]) / sum(self.nsample[-self.period:]) * 100.0
            self.logger.info("Processed {:d} batches"
                             "(loss = {:+.2f}, ce = {:f}, accuracy = {:+.2f})...".format(N, avg, avg_ce, acc))

    def report(self, details=False):
        N = len(self.loss)
        if details:
            sstr = ",".join(map(lambda f: "{:.2f}".format(f), self.loss))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))
            sstr = ",".join(map(lambda f: "{:.4f}".format(f), self.ce))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))
        return {
            "loss": sum(self.loss) / N,
            "ce": sum(self.ce) / N,
            "accuracy": sum(self.ncorrect) / sum(self.nsample) * 100.0,
            "batches": N,
            "cost": self.timer.elapsed()
        }
