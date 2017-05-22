#!/bin/bash
#PBS -q gpu
#PBS -l ngpus=2
#PBS -l ncpus=6
#PBS -l mem=32GB
#PBS -l walltime=20:00:00
module load tensorflow/1.0.1-python3.5
cd /short/cp1/sx6361/dist_rep_sent_doc
python=/home/563/sx6361/.pyenv/versions/3.6.1/bin/python
export PYTHONPATH="/home/563/sx6361/.pyenv/versions/3.6.1/lib/python3.6/site-packages:$PWD"
CUDA_VISIBLE_DEVICES=0 "$python" dist_rep_sent_doc/hyperopt/pvdm.py mean imdb &
P1=$!
CUDA_VISIBLE_DEVICES=1 "$python" dist_rep_sent_doc/hyperopt/pvdm.py mean imdb &
P2=$!
wait $P1 $P2
