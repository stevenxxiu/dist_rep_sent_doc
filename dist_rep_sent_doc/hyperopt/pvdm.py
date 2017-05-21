import argparse
import itertools
import json
import random
import shutil
import subprocess
import sys
import uuid
from contextlib import suppress


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('mode')
    arg_parser.add_argument('dataset')
    args = arg_parser.parse_args()
    all_params = {
        'window_size': list(range(5, 12 + 1)),
        'embedding_size': [50, 100, 150, 200, 250, 300, 350, 400],
        'min_freq': [0, 1, 2, 3, 4, 5],
        'sample': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'train_lr': [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001],
        'val_lr': [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001],
        'train_epoch_size': list(range(1, 50 + 1)),
        'val_epoch_size': list(range(1, 50 + 1)),
    }
    for i in itertools.count(0):
        name = f'results/pvdm_{args.mode}_{args.dataset}_{i:03d}.txt'
        with suppress(FileExistsError), open(name, 'x'), open(name, 'a', encoding='utf-8') as out:
            p = {name: random.choice(values) for name, values in all_params.items()}
            train_path = f'__cache__/tf/{uuid.uuid4()}'
            test_path = f'__cache__/tf/{uuid.uuid4()}'
            subprocess.call([sys.executable, 'dist_rep_sent_doc/main.py', args.dataset, 'val', 'pvdm', json.dumps({
                'mode': args.mode, 'window_size': p['window_size'], 'embedding_size': p['embedding_size'],
                'min_freq': p['min_freq'], 'sample': p['sample'], 'lr': p['train_lr'],
                'batch_size': 2048, 'epoch_size': p['train_epoch_size'],
                'save_path': train_path
            })], stdout=out)
            subprocess.call([sys.executable, 'dist_rep_sent_doc/main.py', args.dataset, 'val', 'pvdm', json.dumps({
                'mode': args.mode, 'window_size': p['window_size'], 'embedding_size': p['embedding_size'],
                'min_freq': p['min_freq'], 'sample': p['sample'], 'lr': p['val_lr'],
                'batch_size': 2048, 'epoch_size': p['val_epoch_size'],
                'save_path': test_path, 'train_path': train_path
            })], stdout=out)
            subprocess.call([sys.executable, 'dist_rep_sent_doc/main.py', args.dataset, 'val', 'nn', json.dumps({
                'train_paths': [train_path], 'test_paths': [test_path],
                'layer_sizes': [2], 'lr': 0.01, 'batch_size': 2048, 'epoch_size': 100
            })], stdout=out)
            shutil.rmtree(train_path)
            shutil.rmtree(test_path)


if __name__ == '__main__':
    main()
