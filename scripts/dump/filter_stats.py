import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm


def get_range(name):
    # name = name.replace('_tfidf', '')
    return list(map(int, os.path.splitext(name)[0].split('-')))


def find_name(names, pos):
    for name in names:
        start, end = get_range(name)
        assert start != end, 'you have self-looping at %s' % name
        if start == pos:
            return name, end
    raise Exception('hdf5 file starting with %d not found.')


def check_dump(args):
    print('checking dir contiguity...')
    names = os.listdir(args.dump_dir)
    pos = args.start
    while pos < args.end:
        name, pos = find_name(names, pos)
    assert pos == args.end, 'reached %d, which is different from the specified end %d' % (pos, args.end)
    print('dir contiguity test passed!')
    print('checking file corruption...')
    pos = args.start
    corrupted_paths = []

    all_count = 0
    thresholds = [0.0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    save_bins = {th: 0 for th in thresholds}
    while pos < args.end:
        name, pos = find_name(names, pos)
        path = os.path.join(args.dump_dir, name)
        with h5py.File(path, 'r') as f:
            print('checking %s...' % path)
            for dk, group in tqdm(f.items()):
                filter_start = group['filter_start'][:] 
                filter_end = group['filter_end'][:] 
                for th in thresholds:
                    start_idxs, = np.where(filter_start > th)
                    end_idxs, = np.where(filter_end > th)
                    num_save_vec = len(set(np.concatenate([start_idxs, end_idxs])))
                    save_bins[th] += num_save_vec
                all_count += len(filter_start)
            # break

    print(all_count)
    print(save_bins)
    comp_rate = {th: f'{save_num/all_count*100:.2f}%' for th, save_num in save_bins.items()}
    print(f'Compression rate: {comp_rate}')
    if len(corrupted_paths) > 0:
        print('following files are corrupted:')
        for path in corrupted_paths:
            print(path)
    else:
        print('file corruption test passed!')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump_dir')
    parser.add_argument('start', type=int)
    parser.add_argument('end', type=int)

    return parser.parse_args()


def main():
    args = get_args()
    check_dump(args)


if __name__ == '__main__':
    main()
