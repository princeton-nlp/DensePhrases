import json
import copy
import os
import argparse


def filter_type(args):
    with open(args.input_file, 'r') as fp:
        from_ = json.load(fp)

    keep_types = ['1', '2', '4']
    with open(args.type2id_file, 'r') as fp:
        type2id = json.load(fp)
        valid_ids = set([id_ for type_ in keep_types for id_ in type2id[type_]])
        
    split = os.path.splitext(os.path.basename(args.input_file))[0]
    assert split in ['train', 'dev', 'test']
    # import pdb; pdb.set_trace()

    to = {'data': []}
    orig_cnt = 0
    filtered_cnt = 0
    for article in from_['data']:
        to_article = {'paragraphs': [], 'title': article['title']}
        for para in article['paragraphs']:
            to_para = copy.deepcopy(para)
            to_para['qas'] = []
            if len(para['qas']) == 0:
                orig_cnt += 1
                filtered_cnt += 1
                continue
            else:
                for qa in para['qas']:
                    orig_cnt += 1
                    if split + '_' + str(qa['id']) in valid_ids:
                        to_para['qas'].append(qa)
                        filtered_cnt += 1
            if len(to_para['qas']) == 0:
                continue
            to_article['paragraphs'].append(to_para)
        if len(to_article['paragraphs']) == 0:
            continue
        to['data'].append(to_article)

    print(f'Filtered from {orig_cnt} to {filtered_cnt}')
    output_file = os.path.join(os.path.dirname(args.input_file), split + f'_{"".join(keep_types)}.json')
    print(f'Saving to {output_file}')
    with open(output_file, 'w') as fp:
        json.dump(to, fp)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('type2id_file')
    return parser.parse_args()


def main():
    args = get_args()
    filter_type(args)


if __name__ == '__main__':
    main()
