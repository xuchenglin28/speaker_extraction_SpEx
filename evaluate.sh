#!/usr/bin/env bash

set -eu

# Usage example: evaluate.sh data/rec/wsj0_2mix/tt_aux60/ data/rec/wsj0_2mix/tt_aux60.scp data/wsj0_2mix/tt_aux60/ref.scp

data_dir=$1
ext_scp=$2
ref_scp=$3

find $data_dir -iname "*.wav" | awk '{split($1,a,"/");split(a[length(a)],b,".wav");printf("%s %s\n",b[1],$1)}' > $ext_scp

python compute_sisdr.py --sep_scp $ext_scp --ref_scp $ref_scp
