import json
import os
import argparse

from tqdm import tqdm
import pdb

def normalize(text):
    return text.lower().replace('_', ' ')


def concat_wikisquad(args):
    names = os.listdir(args.input_dir)
    data = {'data': []}
    for name in tqdm(names):
        from_path = os.path.join(args.input_dir, name)
        with open(from_path, 'r') as fp:
            from_ = json.load(fp)

        for ai, article in enumerate(from_['data']):
            article['id'] = int(name) * 1000 + ai

        articles = []
        for article in from_['data']:
            articles.append(article)

        for article in articles:
            to_article = {'title': article['title'], 'paragraphs': []}
            context = ""
            for para_idx, para in enumerate(article['paragraphs']):
                context = context + " " + para['context']
                if args.min_num_chars <= len(context):
                    to_article['paragraphs'].append({'context': context})
                    context = ""                
                # if the length of the last paragraph is less than min_num_chars, 
                # append it to the previous saving
                elif para_idx == len(article['paragraphs']) -1 :
                    if len(to_article['paragraphs']):
                        previous_context = to_article['paragraphs'][-1]['context']
                        previous_context = previous_context + " " + context
                        to_article['paragraphs'][-1]['context'] = previous_context
                    # if no previous saving exists, create it.
                    else:
                        to_article['paragraphs'].append({'context': context})
            
            data['data'].append(to_article)
            
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for start_idx in range(0, len(data['data']), args.docs_per_file):
        to_path = os.path.join(args.output_dir, str(int(start_idx / args.docs_per_file)).zfill(4))
        cur_data = {'data': data['data'][start_idx:start_idx + args.docs_per_file]}
        with open(to_path, 'w') as fp:
            json.dump(cur_data, fp)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--min_num_chars', default=500, type=int)
    parser.add_argument('--docs_per_file', default=1000, type=int)

    return parser.parse_args()


def main():
    args = get_args()
    concat_wikisquad(args)

if __name__ == '__main__':
    main()