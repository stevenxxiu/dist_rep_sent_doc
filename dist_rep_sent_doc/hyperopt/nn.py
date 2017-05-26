import argparse
import json
import subprocess
import sys
from contextlib import suppress


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset')
    arg_parser.add_argument('name')
    arg_parser.add_argument('max_layer_size', type=int)
    arg_parser.add_argument('train_paths')
    arg_parser.add_argument('test_paths')
    args = arg_parser.parse_args()
    layer_sizes = list(range(2, 10 + 1)) + list(range(10, args.max_layer_size + 1, 10))
    for i, layer_size in enumerate(layer_sizes):
        name = f'results/nn_{args.name}_{args.dataset}_val/run_{i:03d}.txt'
        with suppress(FileExistsError), open(name, 'x'), open(name, 'a', encoding='utf-8') as out:
            subprocess.call([sys.executable, 'dist_rep_sent_doc/main.py', args.dataset, 'val', 'nn', json.dumps({
                'train_paths': json.loads(args.train_paths), 'test_paths': json.loads(args.test_paths),
                'layer_sizes': [layer_size, 2], 'lr': 0.01, 'batch_size': 2048, 'epoch_size': 100
            })], stdout=out)


if __name__ == '__main__':
    main()
