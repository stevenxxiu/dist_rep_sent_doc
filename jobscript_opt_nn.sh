#!/bin/bash
python='/cygdrive/c/Program Files/Python36/python.exe'
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=$(cygpath -d $PWD)
export PYTHONUNBUFFERED=1
"$python" dist_rep_sent_doc/hyperopt/nn.py imdb pvdm_concat 50 \
    '["__cache__/tf/pvdm_concat_imdb_train_val"]' '["__cache__/tf/pvdm_concat_imdb_val"]'
"$python" dist_rep_sent_doc/hyperopt/nn.py imdb pvdm_mean 150 \
    '["__cache__/tf/pvdm_mean_imdb_train_val"]' '["__cache__/tf/pvdm_mean_imdb_val"]'
"$python" dist_rep_sent_doc/hyperopt/nn.py imdb dbow 400 \
    '["__cache__/tf/dbow_imdb_train_val"]' '["__cache__/tf/dbow_imdb_val"]'
