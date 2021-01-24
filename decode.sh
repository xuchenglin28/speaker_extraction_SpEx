#!/usr/bin/env bash

################################################################################################
# Usage example: decode.sh 1 data/wsj0_2mix/tt_aux60 data/rec/wsj0_2mix/tt_aux60 exp/wsj0_2mix #
################################################################################################

set -eu

gpuid=$1
data_dir=$2 # i.e. data/wsj0_2mix/tt_aux60
input=$data_dir/mix.scp # i.e. data/wsj0_2mix/tt_aux60/mix.scp
input_aux=$data_dir/aux.scp # i.e. data/wsj0_2mix/tt_aux60/aux.scp
output_dir=$3 # i.e. data/rec/wsj0_2mix/tt_aux60
checkpoint=$4 # i.e. exp/wsj0_2mix
sample_rate=8000
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

python decode.py --gpu=$gpuid --input $input --input_aux $input_aux --sample_rate $sample_rate --output_dir $output_dir --checkpoint $checkpoint --L1=$L1 --L2=$L2 --L3=$L3 --N=$N --B=$B --O=$O --P=$P --Q=$Q --num_spks=$num_spks --spk_embed_dim=$spk_embed_dim

