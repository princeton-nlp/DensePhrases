import argparse
import os
import subprocess

import h5py
from tqdm import tqdm


def get_size(name):
    a, b = list(map(int, os.path.splitext(name)[0].split('-')))
    return b - a


def bin_names(dir_, names, num_bins):
    names = sorted(names, key=lambda name_: -os.path.getsize(os.path.join(dir_, name_)))
    bins = []
    for name in names:
        if len(bins) < num_bins:
            bins.append([name])
        else:
            smallest_bin = min(bins, key=lambda bin_: sum(get_size(name_) for name_ in bin_))
            smallest_bin.append(name)
    return bins


def run_add_to_index(args):
    def get_cmd(dump_paths, offset_):
        return ["python",
                "build_phrase_index.py",
                f"{args.dump_dir}",
                "add",
                "--fine_quant", "SQ4",
                "--dump_paths", f"{dump_paths}",
                "--offset", f"{offset_}",
                "--num_clusters", f"{args.num_clusters}",
                f"{'--cuda' if args.cuda else ''}"]


    dir_ = os.path.join(args.dump_dir, 'phrase')
    names = os.listdir(dir_)
    bins = bin_names(dir_, names, args.num_gpus)
    offsets = [args.max_num_per_file * each for each in range(len(bins))]

    print('adding with offset:')
    for offset, bin_ in zip(offsets, bins):
        print('%d: %s' % (offset, ','.join(bin_)))

    for kk, (bin_, offset) in enumerate(zip(bins, offsets)):
        if args.start <= kk < args.end:
            print(get_cmd(','.join(bin_), offset))
            subprocess.run(get_cmd(','.join(bin_), offset))
        if args.draft:
            break


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_dir', default='dump/76_dev-1B-c')
    parser.add_argument('--num_cpus', default=4, type=int)
    parser.add_argument('--num_gpus', default=60, type=int)
    parser.add_argument('--mem_size', default=40, type=int, help='mem size in GB')
    parser.add_argument('--num_clusters', default=4096, type=int)
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--max_num_per_file', default=int(1e8), type=int,
                        help='max num per file for setting up good offsets.')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=3, type=int)
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    run_add_to_index(args)


if __name__ == '__main__':
    main()
