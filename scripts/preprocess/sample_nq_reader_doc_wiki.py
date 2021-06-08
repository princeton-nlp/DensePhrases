import json
import glob
import pdb
import argparse
from tqdm import tqdm
import os
import random
import time

def main(args):
    sampling_ratio = args.sampling_ratio
    wiki_dir = args.wiki_dir
    docs_wiki_dir = args.docs_wiki_dir
    output_dir = args.output_dir
    
    # count the number of total words in wikidump
    wiki_file_list = glob.glob(os.path.join(wiki_dir,"*"))
    # num_words_in_wiki = 0
    # for filename in tqdm(wiki_file_list, total=len(wiki_file_list)):
    #     with open(filename,'r') as f:
    #         data = json.load(f)['data']

    #     for doc in data:
    #         for paragraph in doc['paragraphs']:
    #             context = paragraph['context']
    #             num_words_in_wiki += len(context.split(" "))
        
    # print(num_words_in_wiki)

    num_words_in_wiki = 2054581517
    num_sample_words = int(num_words_in_wiki * sampling_ratio)
    
    print("num_words_in_wiki={}".format(num_words_in_wiki))
    
    # count the number of total words in docs_wiki
    docs_wiki_file_list = sorted(glob.glob(os.path.join(docs_wiki_dir,"*")))
    num_words_in_docs_wiki = 0
    docs_wiki_titles = {}
    docs_wikis = []
    for filename in tqdm(docs_wiki_file_list, total=len(docs_wiki_file_list)):
        with open(filename,'r') as f:
            data = json.load(f)['data']

        for doc in data:
            docs_wikis.append(doc)
            docs_wiki_titles[doc['title']] = ""
            for paragraph in doc['paragraphs']:
                context = paragraph['context']
                num_words_in_docs_wiki += len(context.split(" "))
    
    print("num_words_in_docs_wiki={}".format(num_words_in_docs_wiki))
    random.seed(2020)
    i = 0
    while True:
        if num_words_in_docs_wiki > num_sample_words:
            break
        
        # random pick from wiki filelist 
        # start_time = time.time()
        random_wiki_file = random.sample(wiki_file_list, 1)[0]
        # if i % 100 == 0:
        #     print("(1) ", time.time() - start_time)
        
        with open(random_wiki_file,'r') as f:
            data = json.load(f)['data']
        
        # random pick from articles
        # start_time = time.time()
        random_articles = random.sample(data, 100)
        # if i % 100 == 0:
        #     print("(2) ", time.time() - start_time)
            
        # start_time = time.time()
        for random_article in random_articles:
            # if already existing article in docs_wiki, then pass
            if random_article['title'] in docs_wiki_titles:
                continue
            docs_wikis.append(random_article)
            docs_wiki_titles[random_article['title']] = ""
        # if i % 100 == 0:
        #     print("(3) ", time.time() - start_time)
        
        # start_time = time.time()
        for random_article in random_articles:
            for paragraph in random_article['paragraphs']:
                context = paragraph['context']
                num_words_in_docs_wiki += len(context.split(" "))
        # if i % 100 == 0:
        #     print("(4) ", time.time() - start_time)
                
        if i % 100 == 0:
            print("title={} len(docs_wiki_titles)={} ratio={}".format(random_article['title'], len(docs_wiki_titles), num_words_in_docs_wiki/num_words_in_wiki))
        i += 1
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # shuffle docs_wikis for balanced file size
    random.shuffle(docs_wikis)
    
    for i in range(int(len(docs_wikis)/1000) + 1):
        output_file = os.path.join(output_dir, '{:d}'.format(i).zfill(4))
        local_docs_wikis = docs_wikis[i*1000:(i+1)*1000]
        
        output = {
            'data' : local_docs_wikis
        }
    
        # save nq_reader
        with open(output_file,'w') as f:
            json.dump(output, f)    
        
    #     # pdb.set_trace()
        
    # wiki_titles = []
    # wiki_title2paragraphs = {}
    # for filename in tqdm(wiki_file_list, total=len(wiki_file_list)):
    #     with open(filename,'r') as f:
    #         data = json.load(f)['data']
            
    #     for doc in data:
    #         title = doc['title']
    #         wiki_titles.append(title)
    #         paragraph = doc['paragraphs']
    #         wiki_title2paragraphs[title] = paragraph
    #         num_wiki += 1

    # assert len(wiki_title2paragraphs) == num_wiki

    # nq_file_list = glob.glob(os.path.join(nq_dir,"*"))
    # nq_titles = []
    # unmatched_titles = []
    # num_matched = 0
    # num_unmatched = 0
    # for filename in tqdm(nq_file_list, total=len(nq_file_list)):
    #     with open(filename,'r') as f:
    #         data = json.load(f)['data']
            
    #     for doc in data:
    #         title = doc['title']
    #         nq_titles.append(title)
    #         if title in wiki_title2paragraphs and len(wiki_title2paragraphs[title])>0:
    #             doc['paragraphs'] = wiki_title2paragraphs[title]
    #             num_matched += 1
    #         else:
    #             unmatched_titles.append(title)
    #             num_unmatched +=1
            
    #         new_paragraphs = []
    #         for paragraph in doc['paragraphs']:
    #             if ('is_paragraph' in paragraph) and (not paragraph['is_paragraph']):
    #                 continue
                
    #             new_paragraphs.append({
    #                 'context': paragraph['context']
    #             })
    #         doc['paragraphs'] = new_paragraphs
                    
    #     if not os.path.exists(output_dir):
    #         os.mkdir(output_dir)
            
    #     output_path = os.path.join(output_dir,os.path.basename(filename))
    #     output = {
    #         'data': data
    #     }
        
    #     with open(output_path, 'w') as f:
    #         json.dump(output, f, indent=2)
            
    #     with open('unmatched_title_old_dev.txt', 'w') as f:
    #         for title in unmatched_titles:
    #             f.writelines(title)
    #             f.writelines("\n")

    # print("num_matched={} num_unmatched={}".format(num_matched, num_unmatched))
    # print("len(nq_titles)={} len(wiki_titles)={}".format(len(nq_titles), len(wiki_titles)))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--sampling_ratio", type=float, required=True)
    parser.add_argument("--wiki_dir", type=str, required=True)
    parser.add_argument("--docs_wiki_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    main(args)
