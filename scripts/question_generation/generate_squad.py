import spacy
import json
import random
import numpy as np
from tqdm import tqdm
from pipelines import pipeline

nlp_sent = spacy.load("en_core_web_sm")
doc = nlp_sent('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
print([(X.text, X.label_) for X in doc.ents])

# Please train your own model on SQuAD and load as below
nlp = pipeline("multitask-qa-qg", model="t5-large-multi-hl/checkpoint-3500", qg_format="highlight")


data_path = '/home/data/squad/train-v1.1.json'
sample = False
print(f'reading {data_path} with sampling: {sample}')
train_set = json.load(open(data_path))
new_train_set = {'data': []}
cnt = 0
answer_stats = []
bs = 16
tmp_path = data_path.replace('.json', '_qg_t5l35-sqd_tmp.json')
tmp_file = open(tmp_path, 'a')

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

        # Add existing QA pairs
        for qa in paragraph['qas']:
            new_paragraph['qas'].append(qa)
            cnt += 1

        # Get sentences
        sents = [sent for sent in nlp_sent(paragraph['context']).sents]
        qa_pairs = []
        try:
            qa_pairs = nlp(paragraph['context'])
        except Exception as e:
            print('Neural QG error:', paragraph['context'][:50], e)

        ents = [[] for _ in range(len(sents))]
        try:
            for sent_idx, sent in enumerate(sents):
                parse_list = [ent for ent in sent.ents]
                ents[sent_idx] += parse_list
        except Exception as e:
            print('NER error:', sent.text, e)

        cst_qa_pairs = []
        try:
            flat_ents = [e for ent in ents for e in ent]
            qg_examples = nlp._prepare_inputs_for_qg_from_answers_hl(
                [sent.text.strip() for sent in sents], [[e.text for e in ent] for ent in ents]
            )
            qg_inputs = [example['source_text'] for example in qg_examples]
            cst_qs = []
            for i in range(0, len(qg_inputs), bs):
                cst_qs += nlp._generate_questions(qg_inputs[i:i+bs])
            assert len(cst_qs) == len(qg_examples)
            cst_qa_pairs = [{'answer': example['answer'], 'question': que} for example, que in zip(qg_examples, cst_qs)]
        except Exception as e:
            print('Ent QG error:', e)

        orig_len = len(qa_pairs)
        qa_pairs = qa_pairs + cst_qa_pairs
        if len(qa_pairs) == 0:
            print('Skipping as no questions generated for:', sent.text)
            continue
        flat_ents = [None]*orig_len + flat_ents

        q_set = []
        for qa_idx, qa_pair in enumerate(qa_pairs):
            ans = qa_pair['answer']
            que = qa_pair['question']
            if que in q_set:
                continue
            q_set.append(que)
            try:
                if flat_ents[qa_idx] is not None:
                    ans_start = flat_ents[qa_idx][0].idx
                else:
                    ans_start = paragraph['context'].index(ans)
            except Exception as e:
                print('Skipping ans:', ans, e)
                continue
            if ans != paragraph['context'][ans_start:ans_start+len(ans)]:
                print(f'skipping mis-match {ans}')
                continue
            new_paragraph['qas'].append({
                'answers': [{'answer_start': ans_start, 'text': ans}],
                'question': que,
                'id': f'{article["title"]}_p{p_idx}_s{sent_idx}_a{qa_idx}',
            })
            tmp_file.write(
                f'{article["title"]}_p{p_idx}_s{sent_idx}_a{qa_idx}\t{que}\t{ans}\t{ans_start}\n'
            )
            cnt += 1

        if len(qa_pairs) > 0:
            print(qa_pairs[0])
        new_article['paragraphs'].append(new_paragraph)

    new_train_set['data'].append(new_article)

write_path = data_path.replace('.json', '_qg_t5l35-sqd.json')
with open(write_path, 'w') as f:
    json.dump(new_train_set, f)

print(f'writing to {write_path} with {cnt} samples')
