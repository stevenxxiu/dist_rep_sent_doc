#!/bin/bash
#PBS -q gpu
#PBS -l ngpus=2
#PBS -l ncpus=6
#PBS -l mem=16GB
#PBS -l walltime=20:00:00
# module load tensorflow/1.0.1-python3.5
# cd /short/cp1/sx6361/dist_rep_sent_doc
# python=/home/563/sx6361/.pyenv/versions/3.6.1/bin/python
CUDA_VISIBLE_DEVICES=0 "$python" dist_rep_sent_doc/hyperopt/imdb_dbow.py &
P1=$!
CUDA_VISIBLE_DEVICES=1 "$python" dist_rep_sent_doc/hyperopt/imdb_dbow.py &
P2=$!
wait $P1 $P2
