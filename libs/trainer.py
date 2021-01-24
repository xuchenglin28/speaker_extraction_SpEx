#!/usr/bin/env python

import os
import sys
sys.path.append("../")

from itertools import permutations
from collections import defaultdict

import torch as th
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

from utils.load_obj import load_obj
from utils.logger import get_logger
from .reporter import Reporter

class Trainer(object):
    def __init__(self,
                 nnet,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 gpuid=0,
                 lr=0.01,
                 weight_decay=1e-5,
                 clip_norm=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=100,
                 no_impr=6):
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = th.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr

        if os.path.exists(os.path.join(checkpoint, "best.pt.tar")):
            resume = os.path.join(checkpoint, "best.pt.tar")
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))
            # load nnet
            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.device)
            self.optimizer = th.optim.Adam(self.nnet.parameters(), lr=lr, weight_decay=weight_decay)
            self.optimizer.load_state_dict(cpt["optim_state_dict"])
        else:
            self.nnet = nnet.to(self.device)
            self.optimizer = th.optim.Adam(self.nnet.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(cpt, os.path.join(self.checkpoint,
                "{0}.pt.tar".format("best" if best else "last")))

    def save_every_checkpoint(self, idx):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(cpt, os.path.join(self.checkpoint,
                "{0}.pt.tar".format(str(idx))))

    def compute_loss(self, egs):
        raise NotImplementedError

    def train(self, data_loader):
        self.logger.info("Set train mode...")
        self.nnet.train()
        reporter = Reporter(self.logger, period=self.logging_period)

        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.device)

            self.optimizer.zero_grad()
            loss, ce, correct, n = self.compute_loss(egs)
            loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)
            self.optimizer.step()

            reporter.add(loss.item(), ce.item(), correct, n)
        return reporter.report()

    def eval(self, data_loader):
        self.logger.info("Set eval mode...")
        self.nnet.eval()
        reporter = Reporter(self.logger, period=self.logging_period)

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.device)
                loss, ce, correct, n = self.compute_loss(egs)
                reporter.add(loss.item(), ce.item(), correct, n)
        return reporter.report(details=False)

    def run(self, train_loader, dev_loader, num_epochs=50):
        # avoid alloc memory from gpu0
        with th.cuda.device(self.gpuid[0]):
            stats = dict()
            # check if save is OK
            self.save_checkpoint(best=False)
            cv = self.eval(dev_loader)
            best_loss = cv["loss"]
            best_ce = cv["ce"]
            best_accuracy = cv["accuracy"]
            self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}, CE = {:f}, ACCURACY = {:.4f}".format(
                self.cur_epoch, best_loss, best_ce, best_accuracy))
            no_impr = 0
            # make sure not inf
            self.scheduler.best = best_loss
            while self.cur_epoch < num_epochs:
                self.cur_epoch += 1
                cur_lr = self.optimizer.param_groups[0]["lr"]
                stats[
                    "title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                        cur_lr, self.cur_epoch)
                tr = self.train(train_loader)
                stats["tr"] = "train = {:+.4f}, {:f}, {:+.2f}({:.2f}m/{:d})".format(
                    tr["loss"], tr["ce"], tr["accuracy"], tr["cost"], tr["batches"])
                cv = self.eval(dev_loader)
                stats["cv"] = "dev = {:+.4f}, {:f}, {:+.2f}({:.2f}m/{:d})".format(
                    cv["loss"], cv["ce"], cv["accuracy"], cv["cost"], cv["batches"])
                stats["scheduler"] = ""
                if cv["loss"] > best_loss:
                    no_impr += 1
                    stats["scheduler"] = "| no impr, best = {:.4f}".format(
                        self.scheduler.best)
                else:
                    best_loss = cv["loss"]
                    no_impr = 0
                    self.save_checkpoint(best=True)
                self.save_every_checkpoint(self.cur_epoch)
                self.logger.info(
                    "{title} {tr} | {cv} {scheduler}".format(**stats))
                # schedule here
                self.scheduler.step(cv["loss"])
                # flush scheduler info
                sys.stdout.flush()
                # save last checkpoint
                self.save_checkpoint(best=False)
                if no_impr == self.no_impr:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_impr))
                    break
            self.logger.info("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, num_epochs))


class SiSnrTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SiSnrTrainer, self).__init__(*args, **kwargs)

    def sisdr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        sisdr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return th.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - th.mean(x, dim=-1, keepdim=True)
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        t = th.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * th.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def mask_by_length(self, xs, lengths, fill=0):
        """Mask tensor according to length.

        Args:
            xs (Tensor): Batch of input tensor (B, `*`).
            lengths (LongTensor or List): Batch of lengths (B,).
            fill (int or float): Value to fill masked part.

        Returns:
            Tensor: Batch of masked input tensor (B, `*`).

        Examples:
            >>> x = torch.arange(5).repeat(3, 1) + 1
            >>> x
            tensor([[1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5]])
            >>> lengths = [5, 3, 2]
            >>> mask_by_length(x, lengths)
            tensor([[1, 2, 3, 4, 5],
                    [1, 2, 3, 0, 0],
                    [1, 2, 0, 0, 0]])

        """
        assert xs.size(0) == len(lengths)
        ret = xs.data.new(*xs.size()).fill_(fill)
        for i, l in enumerate(lengths):
            ret[i, :l] = xs[i, :l]
        return ret

    def compute_loss(self, egs):
        ests, ests2, ests3, spk_pred = th.nn.parallel.data_parallel(
            self.nnet, (egs["mix"], egs["aux"], egs["aux_len"]), device_ids=self.gpuid)
        refs = egs["ref"]

        ## P x N
        N = egs["mix"].size(0)
        valid_len = egs["valid_len"]
        ests = self.mask_by_length(ests, valid_len)
        ests2 = self.mask_by_length(ests2, valid_len)
        ests3 = self.mask_by_length(ests3, valid_len)
        refs = self.mask_by_length(refs, valid_len)

        snr1 = self.sisdr(ests, refs)
        snr2 = self.sisdr(ests2, refs)
        snr3 = self.sisdr(ests3, refs)
        snr_loss = (-0.8*th.sum(snr1)-0.1*th.sum(snr2)-0.1*th.sum(snr3)) / N
 
        ce = th.nn.CrossEntropyLoss()
        ce_loss = ce(spk_pred, egs["spk_idx"])

        # calculate accuracy
        probs = th.softmax(spk_pred, dim=1)
        est_label = probs.argmax(dim=1)
        correct = (est_label==egs["spk_idx"]).sum().item()

        return snr_loss + 10 * ce_loss, ce_loss, correct, N
