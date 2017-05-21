import itertools
import json
import random
import subprocess
import sys
import argparse
from contextlib import suppress


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset')
    args = arg_parser.parse_args()
    name_part = args.dataset
    all_params = {
        'embedding_size': [50, 100, 150, 200, 250, 300, 350, 400],
        'min_freq': [0, 1, 2, 3, 4, 5],
        'sample': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'train_lr': [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001],
        'val_lr': [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001],
        'train_epoch_size': list(range(1, 50 + 1)),
        'val_epoch_size': list(range(1, 50 + 1)),
    }
    for i in itertools.count(0):
        name = f'results/dbow_{name_part}_{i:03d}.txt'
        with suppress(FileExistsError), open(name, 'x'), open(name, 'a', encoding='utf-8') as out:
            p = {name: random.choice(values) for name, values in all_params.items()}
            subprocess.call([sys.executable, 'dist_rep_sent_doc/main.py', args.dataset, 'val', 'dbow', json.dumps({
                'embedding_size': p['embedding_size'], 'min_freq': p['min_freq'], 'sample': p['sample'],
                'lr': p['train_lr'], 'batch_size': 2048, 'epoch_size': p['train_epoch_size'],
                'save_path': f'__cache__/tf/dbow_{name_part}_train'
            })], stdout=out)
            subprocess.call([sys.executable, 'dist_rep_sent_doc/main.py', args.dataset, 'val', 'dbow', json.dumps({
                'embedding_size': p['embedding_size'], 'min_freq': p['min_freq'], 'sample': p['sample'],
                'lr': p['val_lr'], 'batch_size': 2048, 'epoch_size': p['val_epoch_size'],
                'save_path': f'__cache__/tf/dbow_{name_part}_val',
                'train_path': f'__cache__/tf/dbow_{name_part}_train'
            })], stdout=out)
            subprocess.call([sys.executable, 'dist_rep_sent_doc/main.py', args.dataset, 'val', 'nn', json.dumps({
                'train_paths': [f'__cache__/tf/dbow_{name_part}_train'],
                'test_paths': [f'__cache__/tf/dbow_{name_part}_val'],
                'layer_sizes': [2], 'lr': 0.01, 'batch_size': 2048, 'epoch_size': 100
            })], stdout=out)


if __name__ == '__main__':
    main()