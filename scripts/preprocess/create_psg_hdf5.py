import json
import argparse
import os
import h5py
import csv

from tqdm import tqdm


def create_psg_hdf5(input_file, out_file):
    passages = {}
    with open(input_file) as f:
        psg_file = csv.reader(f, delimiter='\t')
        for data_idx, data in tqdm(enumerate(psg_file)):
            if data_idx == 0:
                print('Reading', data)
                continue
            id_, psg, title = data
            passages[id_] = [psg, title]
            # break

    # Must use bucket; otherwise writing to a hdf5 file is very slow with a large number of keys
    bucket_size = 1000000
    # buckets = [(start, min(start+bucket_size-1, 21015324)) for start in range(1, 21015325, bucket_size)]
    buckets = [(start, min(start+bucket_size-1, len(passages))) for start in range(1, len(passages)+1, bucket_size)]
    print(f'Putting {len(passages)} passages into {len(buckets)} buckets')
    print(buckets)
    with h5py.File(out_file, 'w') as f:
        for pid, data in tqdm(passages.items()):
            bucket_name = None
            for start, end in buckets:
                if (int(pid) >= start) and (int(pid) <= end):
                    bucket_name = f'{start}-{end}'
                    break
            assert bucket_name is not None
            # continue

            if bucket_name not in f:
                dg = f.create_group(bucket_name)
            else:
                dg = f[bucket_name]
            assert pid not in dg
            pg = dg.create_group(pid)
            pg.attrs['context'], pg.attrs['title'] = data

    print(f'Saving {out_file} done')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, default=None)
    parser.add_argument('out_file', type=str)
    args = parser.parse_args()
    create_psg_hdf5(args.input_file, args.out_file)
