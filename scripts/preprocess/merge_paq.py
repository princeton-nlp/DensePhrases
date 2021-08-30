import json
import argparse
import os
import h5py
import csv

from tqdm import tqdm


def merge_paq(input_dir, out_file):
    num_split = 8
    filenames = [f'PAQ.metadata.hard0-{k}.jsonl' for k in range(num_split)]
    print('reading', filenames)
    fps = [open(os.path.join(input_dir, filename), 'r') for filename in filenames]
    
    with open(out_file, 'w') as fw:
        fp_idx = 0  
        total_cnt = 0
        hard_cnt = 0
        line = fps[fp_idx].readline()
        while line:
            # for stats
            meta = json.loads(line)
            if len(meta['hard_neg_pids']) > 0:
                hard_cnt += 1
            total_cnt += 1
            
            if total_cnt % 100000 == 0:
                print(f'Total: {total_cnt}, Hard neg: {hard_cnt}')

            # write it
            json.dump(meta, fw, separators=(',', ':'))
            fw.write('\n')
            fp_idx = (fp_idx + 1) % num_split
            line = fps[fp_idx].readline()

    print(f'Total: {total_cnt}, Hard neg: {hard_cnt}')
    print(f'Saving {out_file} done')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, default=None)
    parser.add_argument('out_file', type=str)
    args = parser.parse_args()
    merge_paq(args.input_dir, args.out_file)
