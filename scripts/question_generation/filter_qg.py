import spacy
import json
import random
import numpy as np
from tqdm import tqdm
from squad_metrics import compute_exact
nlp = spacy.load("en_core_web_sm")

doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
print([(X.text, X.label_) for X in doc.ents])


data_path = 'data/squad-nq/train-sqdqg_nqqg.json'
sample = False
print(f'reading {data_path} with sampling: {sample}')
train_set = json.load(open(data_path))
new_train_set = {'data': []}
cnt = 0
new_cnt = 0
orig_cnt = 0
miss_cnt = 0

prediction_path = 'models/spanbert-base-cased-sqdnq_qgfilter/predictions_.json'
predictions = {str(id_): pred for id_, pred in json.load(open(prediction_path)).items()}

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
            id_ = str(qa['id'])
            # assert id_ in predictions
            if id_ not in predictions:
                print('missing predictions', id_)
                miss_cnt += 1
                continue
            if all(kk in id_ for kk in['_p', '_s', '_a']):
                if not compute_exact(qa['answers'][0]['text'], predictions[id_]):
                    continue
                else:
                    new_cnt += 1
            else:
                orig_cnt += 1

            new_paragraph['qas'].append(qa)
            cnt += 1
        new_article['paragraphs'].append(new_paragraph)

    new_train_set['data'].append(new_article)
    # break

write_path = data_path.replace('.json', '_filtered.json')
with open(write_path, 'w') as f:
    json.dump(new_train_set, f)

assert orig_cnt + new_cnt == cnt
print(f'writing to {write_path} with {cnt} samples')
print(f'orig sample: {orig_cnt}, new sample: {new_cnt}')
print(f'missing sample: {miss_cnt}')
