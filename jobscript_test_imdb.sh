#!/bin/bash
python='/cygdrive/c/Program Files/Python36/python.exe'
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=$(cygpath -d $PWD)
export PYTHONUNBUFFERED=1

"$python" dist_rep_sent_doc/main.py imdb test pvdm '{
    "mode": "mean", "window_size": 8, "embedding_size": 150,
    "min_freq": 4, "sample": 0.001, "lr": 0.0025, "batch_size": 2048, "epoch_size": 26,
    "save_path": "__cache__/tf/pvdm_mean_imdb_train_test"
}'
"$python" dist_rep_sent_doc/main.py imdb test pvdm '{
    "mode": "mean", "window_size": 8, "embedding_size": 150,
    "min_freq": 4, "sample": 0.001, "lr": 0.0001, "batch_size": 2048, "epoch_size": 49,
    "save_path": "__cache__/tf/pvdm_mean_imdb_test", "train_path": "__cache__/tf/pvdm_mean_imdb_train_test"
}'
"$python" dist_rep_sent_doc/main.py imdb test pvdm '{
    "mode": "concat", "window_size": 7, "embedding_size": 50,
    "min_freq": 2, "sample": 0.001, "lr": 0.0005, "batch_size": 2048, "epoch_size": 46,
    "save_path": "__cache__/tf/pvdm_concat_imdb_train_test"
}'
"$python" dist_rep_sent_doc/main.py imdb test pvdm '{
    "mode": "concat", "window_size": 7, "embedding_size": 50,
    "min_freq": 2, "sample": 0.001, "lr": 0.00025, "batch_size": 2048, "epoch_size": 42,
    "save_path": "__cache__/tf/pvdm_concat_imdb_test", "train_path": "__cache__/tf/pvdm_concat_imdb_train_test"
}'
"$python" dist_rep_sent_doc/main.py imdb test dbow '{
    "embedding_size": 400, "min_freq": 0, "sample": 0.1, "lr": 0.001, "batch_size": 2048, "epoch_size": 12,
    "save_path": "__cache__/tf/dbow_imdb_train_test"
}'
"$python" dist_rep_sent_doc/main.py imdb test dbow '{
    "embedding_size": 400, "min_freq": 0, "sample": 0.1, "lr": 0.0001, "batch_size": 2048, "epoch_size": 18,
    "save_path": "__cache__/tf/dbow_imdb_test", "train_path": "__cache__/tf/dbow_imdb_train_test"
}'
"$python" dist_rep_sent_doc/main.py imdb test dbow '{
    "sg": 4, "embedding_size": 400, "min_freq": 0, "sample": 0.1, "lr": 0.001, "batch_size": 2048, "epoch_size": 12,
    "save_path": "__cache__/tf/dbow_sg_imdb_train_test"
}'
"$python" dist_rep_sent_doc/main.py imdb test dbow '{
    "sg": 4, "embedding_size": 400, "min_freq": 0, "sample": 0.1, "lr": 0.0001, "batch_size": 2048, "epoch_size": 18,
    "save_path": "__cache__/tf/dbow_sg_imdb_test", "train_path": "__cache__/tf/dbow_sg_imdb_train_test"
}'
"$python" dist_rep_sent_doc/main.py imdb test nn '{
    "train_paths": ["__cache__/tf/dbow_imdb_train_test"],
    "test_paths": ["__cache__/tf/dbow_imdb_test"],
    "layer_sizes": [2], "lr": 0.01, "batch_size": 2048, "epoch_size": 100
}'
