import json
import pdb

data_for_denspi = []
data_for_dpr = []

with open('benchmark/nq_1000_dev_orqa.jsonl', encoding='utf-8') as f:
    idx = 0
    while True:
        line = f.readline()
        if line == "":
            break

        sample = json.loads(line)
        
        data_for_denspi.append({
            'id':f'dev_{idx}',
            'question': sample['question'],
            'answers': sample['answer']
        })
        data_for_dpr.append("\t".join([sample['question'], str(sample['answer'])]))
        
        idx += 1

# save data_for_dpr as csv
with open('benchmark/nq_1000_dev_dpr.csv', 'w', encoding='utf-8') as f:
    for line in data_for_dpr:
        f.writelines(line)
        f.writelines("\n")

# save data_for_denspi as json
with open('benchmark/nq_1000_dev_denspi.json', 'w', encoding='utf-8') as f:
    json.dump({'data': data_for_denspi}, f)
