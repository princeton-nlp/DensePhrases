import spacy
import json
import random
import numpy as np
from tqdm import tqdm
from squad_metrics import compute_exact
nlp = spacy.load("en_core_web_sm")

doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
print([(X.text, X.label_) for X in doc.ents])


data_path = '/home/data/nq-reader/dev_wiki3.json'
sample = False
print(f'reading {data_path} with sampling: {sample}')
train_set = json.load(open(data_path))
new_train_set = {'data': []}
cnt = 0
new_cnt = 0
filtered_cnt = 0

for article in tqdm(train_set['data']):
    new_article = {
        'title': article['title'],
        'paragraphs': []
    }
    for p_idx, paragraph in enumerate(article['paragraphs']):
        new_paragraph = {
            'context': paragraph['context'],
            'qas' : [],
        }

        for qa in paragraph['qas']:
            question = qa['question']
            id_ = qa['id']
            assert type(qa["answers"]) == dict or type(qa["answers"]) == list, type(qa["answers"])
            if type(qa["answers"]) == dict:
                qa["answers"] = [qa["answers"]]
            cnt += 1
            if len(qa["answers"]) == 0:
                filtered_cnt += 1
                continue

            new_paragraph['qas'].append(qa)
            new_cnt += 1
        new_article['paragraphs'].append(new_paragraph)

    new_train_set['data'].append(new_article)
    # break

write_path = data_path.replace('.json', '_na_filtered.json')
with open(write_path, 'w') as f:
    json.dump(new_train_set, f)

assert filtered_cnt + new_cnt == cnt
print(f'writing to {write_path} with {cnt} samples')
print(f'all sample: {cnt}, new sample: {new_cnt}')
