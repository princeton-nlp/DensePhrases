import argparse
import os
import h5py
import torch
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
    metadata = {}
    keys_to_save = ['f2o_end', 'f2o_start', 'span_logits', 'start2end', 'word2char_end', 'word2char_start']
    while pos < args.end:
        name, pos = find_name(names, pos)
        path = os.path.join(args.dump_dir, name)
        try:
            with h5py.File(path, 'r') as f:
                print('checking %s...' % path)
                for dk, group in tqdm(f.items()):
                    # keys = list(group.keys())
                    metadata[dk] = {save_key: group[save_key][:] for save_key in keys_to_save}
                    metadata[dk]['context'] = group.attrs['context']
                    metadata[dk]['title'] = group.attrs['title']
        except Exception as e:
            print(e)
            print('%s corrupted!' % path)
            corrupted_paths.append(path)

        break
    
    torch.save(metadata, 'tmp.bin')
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
