#!/bin/bash
#PBS -q gpupascal
#PBS -l ngpus=1
#PBS -l ncpus=6
#PBS -l mem=16GB
#PBS -l walltime=20:00:00
module load tensorflow/1.0.1-python3.5
cd /short/cp1/sx6361/dist_rep_sent_doc
python=/home/563/sx6361/.pyenv/versions/3.6.1/bin/python
export PYTHONPATH="/home/563/sx6361/.pyenv/versions/3.6.1/lib/python3.6/site-packages:$PWD"

"$python" dist_rep_sent_doc/main.py imdb test pvdm '{
    "mode": "concat", "window_size": 9, "embedding_size": 100,
    "min_freq": 2, "sample": 1e-3, "lr": 0.001, "batch_size": 2048, "epoch_size": 20,
    "save_path": "__cache__/tf/pvdm_concat_imdb_train"
}'
"$python" dist_rep_sent_doc/main.py imdb test pvdm '{
    "mode": "concat", "window_size": 9, "embedding_size": 100,
    "min_freq": 2, "sample": 1e-3, "lr": 0.001, "batch_size": 2048, "epoch_size": 10,
    "save_path": "__cache__/tf/pvdm_concat_imdb_test", "train_path": "__cache__/tf/pvdm_concat_imdb_train"
}'
"$python" dist_rep_sent_doc/main.py imdb test pvdm '{
    "mode": "mean", "window_size": 9, "embedding_size": 100,
    "min_freq": 2, "sample": 1e-3, "lr": 0.001, "batch_size": 2048, "epoch_size": 20,
    "save_path": "__cache__/tf/pvdm_mean_imdb_train"
}'
"$python" dist_rep_sent_doc/main.py imdb test pvdm '{
    "mode": "mean", "window_size": 9, "embedding_size": 100,
    "min_freq": 2, "sample": 1e-3, "lr": 0.001, "batch_size": 2048, "epoch_size": 10,
    "save_path": "__cache__/tf/pvdm_mean_imdb_test", "train_path": "__cache__/tf/pvdm_mean_imdb_train"
}'
"$python" dist_rep_sent_doc/main.py imdb test dbow '{
    "embedding_size": 100, "min_freq": 2, "sample": 1e-3, "lr": 0.001, "batch_size": 2048, "epoch_size": 20,
    "save_path": "__cache__/tf/dbow_imdb_train"
}'
"$python" dist_rep_sent_doc/main.py imdb test dbow '{
    "embedding_size": 100, "min_freq": 2, "sample": 1e-3, "lr": 0.001, "batch_size": 2048, "epoch_size": 10,
    "save_path": "__cache__/tf/dbow_imdb_test", "train_path": "__cache__/tf/dbow_imdb_train"
}'
"$python" dist_rep_sent_doc/main.py imdb test nn '{
    "train_paths": ["__cache__/tf/pvdm_concat_imdb_train", "__cache__/tf/dbow_imdb_train"],
    "test_paths": ["__cache__/tf/pvdm_concat_imdb_test", "__cache__/tf/dbow_imdb_test"],
    "layer_sizes": [2], "lr": 0.01, "batch_size": 2048, "epoch_size": 100
}'
