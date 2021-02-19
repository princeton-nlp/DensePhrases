import json
import argparse
import os

from tqdm import tqdm


def merge_openqa(input_dir, output_path):

    paths = [
        'open-qa/nq-open/train_preprocessed.json',
        'open-qa/webq/WebQuestions-train-nodev_preprocessed.json',
        'open-qa/trec/CuratedTrec-train-nodev_preprocessed.json',
        'open-qa/triviaqa-unfiltered/train_preprocessed.json',
        'open-qa/squad/train_preprocessed.json',
        'kilt/trex/trex-train-kilt_open_10000.json',
        'kilt/zsre/structured_zeroshot-train-kilt_open_10000.json',
    ]
    paths = [os.path.join(input_dir, path) for path in paths]
    assert all([os.path.exists(path) for path in paths])

    data_to_save = []
    sep_cnt = 0
    for path in paths:
        with open(path) as f:
            data = json.load(f)['data']
            for item in data:
                if ' [SEP] ' in item['question']:
                    item['question'] = item['question'].replace(' [SEP] ', ' ')
                    sep_cnt += 1
            data_to_save += data
        print(f'{path} has {len(data)} QA pairs')

    print(f'Saving {len(data_to_save)} questions to output_path')
    print(f'Removed [SEP] for {sep_cnt} questions')
    print('Writing to %s\n'% output_path)
    with open(output_path, 'w') as f:
        json.dump({'data': data_to_save}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, default=None)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    merge_openqa(args.input_dir, args.output_path)
