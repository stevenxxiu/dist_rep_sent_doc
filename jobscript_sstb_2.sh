#!/bin/bash
#PBS -q gpupascal
#PBS -l ngpus=1
#PBS -l ncpus=6
#PBS -l mem=16GB
#PBS -l walltime=20:00:00
module load tensorflow/1.0.1-python3.5
cd /short/cp1/sx6361/dist_rep_sent_doc
python=/home/563/sx6361/.pyenv/versions/3.6.1/bin/python

"$python" dist_rep_sent_doc/main.py sstb_2 pvdm '{
    "mode": "concat", "window_size": 8, "embedding_size": 100,
    "min_freq": 2, "sample": 1e-3, "lr": 0.01, "batch_size": 2048, "epoch_size": 30,
    "save_path": "__cache__/tf/pvdm_concat_sstb_2_train"
}'
"$python" dist_rep_sent_doc/main.py sstb_2 pvdm '{
    "mode": "concat", "window_size": 8, "embedding_size": 100,
    "min_freq": 2, "sample": 1e-3, "lr": 0.1, "batch_size": 2048, "epoch_size": 20,
    "save_path": "__cache__/tf/pvdm_concat_sstb_2_test", "train_path": "__cache__/tf/pvdm_concat_sstb_2_train"
}'
"$python" dist_rep_sent_doc/main.py sstb_2 dbow '{
    "embedding_size": 100, "min_freq": 2, "sample": 1e-3, "lr": 0.01, batch_size: 2048, epoch_size: 30,
    "save_path": "__cache__/tf/dbow_sstb_2_train"
}'
"$python" dist_rep_sent_doc/main.py sstb_2 dbow '{
    "embedding_size": 100, "min_freq": 2, "sample": 1e-3, "lr": 0.1, batch_size: 2048, epoch_size: 20,
    "save_path": "__cache__/tf/dbow_sstb_2_test", "train_path": "__cache__/tf/dbow_sstb_2_train"
}'
"$python" dist_rep_sent_doc/main.py sstb_2 nn '{
    "train_paths": ["__cache__/tf/pvdm_concat_sstb_2_train", "__cache__/tf/dbow_sstb_2_train"],
    "test_paths": ["__cache__/tf/pvdm_concat_sstb_2_test", "__cache__/tf/dbow_sstb_2_test"],
    "layer_sizes": [2], "lr": 0.001, "batch_size": 2048, "epoch_size": 20
}'
