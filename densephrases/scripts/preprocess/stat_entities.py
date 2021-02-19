import spacy
import json
import random
import numpy as np
from tqdm import tqdm

nlp_sent = spacy.load("en_core_web_sm")
doc = nlp_sent('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
print([(X.text, X.label_) for X in doc.ents])


# pred_file = '/n/fs/nlp-jl5167/outputs/pred/dev_preprocessed_8757.pred'
pred_file = 'lama-test-P20_preprocessed_953.pred'
with open(pred_file) as f:
    predictions = json.load(f)

stat = {}
ent_types = {}
tokenizer_error_cnt = 0
entity_error_cnt = 0
for pid, result in predictions.items():
    question = result['question']
    q_sws = result['q_tokens'][1:-1] # except [CLS], [SEP]
    q_ents = [(X.text, X.label_, X[0].idx) for X in nlp_sent(question).ents]
    if len(q_ents) == 0:
        entity_error_cnt += 1
        continue

    word_idx = 0
    word_to_sw = {}
    for sw_idx, sw in enumerate(q_sws): 
        if word_idx not in word_to_sw:
            word_to_sw[word_idx] = []
        word_to_sw[word_idx].append(sw_idx)
        if sw_idx < len(q_sws) - 1:
            if not q_sws[sw_idx+1].startswith('##'):
                word_idx += 1
    try:    
        assert word_idx == len(question.split(' ')) - 1
    except Exception as e:
        tokenizer_error_cnt += 1
        continue

    char_to_word = {}
    word_idx = 0
    for ch_idx, ch in enumerate(question):
        if ch == ' ':
            word_idx += 1
            continue
        char_to_word[ch_idx] = word_idx

    try:
        assert word_idx == len(question.split(' ')) - 1
    except Exception as e:
        tokenizer_error_cnt += 1
        continue

    num_sw = []
    ent_list = [
        'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC',
        'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART'
    ]
    for ent_text, ent_label, ent_start in q_ents:
        if ent_label not in ent_list:
            continue
        char_start = ent_start
        char_end = ent_start + len(ent_text) - 1
        word_start = char_to_word[char_start]
        word_end = char_to_word[char_end]
        num_sw.append(sum([len(word_to_sw[word]) for word in range(word_start, word_end+1)]))
        # num_sw.append(max([len(word_to_sw[word]) for word in range(word_start, word_end+1)]))
        if ent_label not in ent_types:
            ent_types[ent_label] = 0
            print(ent_text, ent_label)
        ent_types[ent_label] += 1

    if len(num_sw) == 0:
        entity_error_cnt += 1
        continue

    num_sw = max(num_sw)
    if num_sw not in stat:
        print(num_sw, q_sws)
        stat[num_sw] = []
    stat[num_sw].append(int(result['em_top1']))

output = sorted({key: (f'{sum(val)/len(val):.2f}', f'{len(val)} Qs') for key, val in stat.items()}.items())
print(f'exclude {tokenizer_error_cnt} questions for tokenization error')
print(f'exclude {entity_error_cnt} questions for entity not found error')
print(f'stat: {output} for {len(predictions) - tokenizer_error_cnt - entity_error_cnt} questions')
print(sorted(ent_types.items()))
