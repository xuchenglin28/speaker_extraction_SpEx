#!/usr/bin/env python

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .norm import ChannelwiseLayerNorm, GlobalLayerNorm
from .cnns import Conv1D, ConvTrans1D, TCNBlock, TCNBlock_Spk, ResBlock

class SpEx_Plus(nn.Module):
    def __init__(self,
                 L1=20,
                 L2=80,
                 L3=160,
                 N=256,
                 B=8,
                 O=256,
                 P=512,
                 Q=3,
                 num_spks=101,
                 spk_embed_dim=256,
                 causal=False):
        super(SpEx_Plus, self).__init__()
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.encoder_1d_short = Conv1D(1, N, L1, stride=L1 // 2, padding=0)
        self.encoder_1d_middle = Conv1D(1, N, L2, stride=L1 // 2, padding=0)
        self.encoder_1d_long = Conv1D(1, N, L3, stride=L1 // 2, padding=0)
        # before repeat blocks, always cLN
        self.ln = ChannelwiseLayerNorm(3*N)
        # n x N x T => n x O x T
        self.proj = Conv1D(3*N, O, 1)
        self.conv_block_1 = TCNBlock_Spk(spk_embed_dim=spk_embed_dim, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal, dilation=1)
        self.conv_block_1_other = self._build_stacks(num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal)
        self.conv_block_2 = TCNBlock_Spk(spk_embed_dim=spk_embed_dim, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal, dilation=1)
        self.conv_block_2_other = self._build_stacks(num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal)
        self.conv_block_3 = TCNBlock_Spk(spk_embed_dim=spk_embed_dim, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal, dilation=1)
        self.conv_block_3_other = self._build_stacks(num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal)
        self.conv_block_4 = TCNBlock_Spk(spk_embed_dim=spk_embed_dim, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal, dilation=1)
        self.conv_block_4_other = self._build_stacks(num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal)
        # n x O x T => n x N x T
        self.mask1 = Conv1D(O, N, 1)
        self.mask2 = Conv1D(O, N, 1)
        self.mask3 = Conv1D(O, N, 1)
        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        self.decoder_1d_short = ConvTrans1D(N, 1, kernel_size=L1, stride=L1 // 2, bias=True)
        self.decoder_1d_middle = ConvTrans1D(N, 1, kernel_size=L2, stride=L1 // 2, bias=True)
        self.decoder_1d_long = ConvTrans1D(N, 1, kernel_size=L3, stride=L1 // 2, bias=True)
        self.num_spks = num_spks

        self.spk_encoder = nn.Sequential(
            ChannelwiseLayerNorm(3*N),
            Conv1D(3*N, O, 1),
            ResBlock(O, O),
            ResBlock(O, P),
            ResBlock(P, P),
            Conv1D(P, spk_embed_dim, 1),
        )

        self.pred_linear = nn.Linear(spk_embed_dim, num_spks)

    def _build_stacks(self, num_blocks, **block_kwargs):
        """
        Stack B numbers of TCN block, the first TCN block takes the speaker embedding
        """
        blocks = [
            TCNBlock(**block_kwargs, dilation=(2**b))
            for b in range(1,num_blocks)
        ]
        return nn.Sequential(*blocks)

    def forward(self, x, aux, aux_len):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        # when inference, only one utt
        if x.dim() == 1:
            x = th.unsqueeze(x, 0)

        # n x 1 x S => n x N x T
        w1 = F.relu(self.encoder_1d_short(x))
        T = w1.shape[-1]
        xlen1 = x.shape[-1]
        xlen2 = (T - 1) * (self.L1 // 2) + self.L2
        xlen3 = (T - 1) * (self.L1 // 2) + self.L3
        w2 = F.relu(self.encoder_1d_middle(F.pad(x, (0, xlen2 - xlen1), "constant", 0)))
        w3 = F.relu(self.encoder_1d_long(F.pad(x, (0, xlen3 - xlen1), "constant", 0)))

        # n x 3N x T
        y = self.ln(th.cat([w1, w2, w3], 1))
        # n x O x T
        y = self.proj(y)
        
        # speaker encoder (share params from speech encoder)
        aux_w1 = F.relu(self.encoder_1d_short(aux))
        aux_T_shape = aux_w1.shape[-1]
        aux_len1 = aux.shape[-1]
        aux_len2 = (aux_T_shape - 1) * (self.L1 // 2) + self.L2
        aux_len3 = (aux_T_shape - 1) * (self.L1 // 2) + self.L3
        aux_w2 = F.relu(self.encoder_1d_middle(F.pad(aux, (0, aux_len2 - aux_len1), "constant", 0)))
        aux_w3 = F.relu(self.encoder_1d_long(F.pad(aux, (0, aux_len3 - aux_len1), "constant", 0)))

        aux = self.spk_encoder(th.cat([aux_w1, aux_w2, aux_w3], 1))
        aux_T = (aux_len - self.L1) // (self.L1 // 2) + 1
        aux_T = ((aux_T // 3) // 3) // 3
        aux = th.sum(aux, -1)/aux_T.view(-1,1).float()

        y = self.conv_block_1(y, aux)
        y = self.conv_block_1_other(y)
        y = self.conv_block_2(y, aux)
        y = self.conv_block_2_other(y)
        y = self.conv_block_3(y, aux)
        y = self.conv_block_3_other(y)
        y = self.conv_block_4(y, aux)
        y = self.conv_block_4_other(y)

        # n x N x T
        m1 = F.relu(self.mask1(y))
        m2 = F.relu(self.mask2(y))
        m3 = F.relu(self.mask3(y))
        S1 = w1 * m1
        S2 = w2 * m2
        S3 = w3 * m3

        return self.decoder_1d_short(S1), self.decoder_1d_middle(S2)[:, :xlen1], self.decoder_1d_long(S3)[:, :xlen1], self.pred_linear(aux)
