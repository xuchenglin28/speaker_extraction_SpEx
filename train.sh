#!/usr/bin/env bash

# Usage example: bash train.sh 0,1,2 exp/wsj0_2mix 36

set -eu

gpuid=$1
cpt_dir=$2
batch_size=$3 # 36 for 3 32G V100, constrainted by GPU number & memory
epochs=150
L1=0.0025
L2=0.01
L3=0.02
N=256
B=8
O=256
P=512
Q=3
num_spks=101
spk_embed_dim=256
train_dir=data/wsj0_2mix/tr
dev_dir=data/wsj0_2mix/cv
spk_list=data/wsj0_2mix_extr_tr.spk
sample_rate=8000
lr=0.001

python -u train.py --gpu $gpuid --epochs $epochs --batch-size $batch_size --checkpoint $cpt_dir \
    --L1=$L1 --L2=$L2 --L3=$L3 --N=$N --B=$B --O=$O --P=$P --Q=$Q --num_spks=$num_spks \
    --spk_embed_dim=$spk_embed_dim --train_dir=$train_dir --dev_dir=$dev_dir --spk_list=$spk_list \
    --sample_rate=$sample_rate --lr=$lr
