import json
import argparse
import torch
import os
import random
import numpy as np
import requests
import logging
import math
import copy
import string
import faiss
import csv

from time import time
from tqdm import tqdm

from densephrases.utils.eval_utils import normalize_answer, f1_score, exact_match_score, drqa_exact_match_score, \
        drqa_regex_match_score, drqa_metric_max_over_ground_truths, drqa_normalize
from densephrases.utils.open_utils import load_query_encoder, load_phrase_index, get_query2vec, load_qa_pairs
from densephrases.utils.kilt.eval import evaluate as kilt_evaluate
from densephrases.utils.kilt.kilt_utils import store_data as kilt_store_data

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def embed_all_query(questions, args, query_encoder, tokenizer, batch_size=64):
    query2vec = get_query2vec(
        query_encoder=query_encoder, tokenizer=tokenizer, args=args, batch_size=batch_size
    )

    all_outs = []
    for q_idx in tqdm(range(0, len(questions), batch_size)):
        outs = query2vec(questions[q_idx:q_idx+batch_size])
        all_outs += outs
    start = np.concatenate([out[0] for out in all_outs], 0)
    end = np.concatenate([out[1] for out in all_outs], 0)
    query_vec = np.concatenate([start, end], 1)
    logger.info(f'Query reps: {query_vec.shape}')
    return query_vec


def evaluate(args, mips=None, query_encoder=None, tokenizer=None, q_idx=None):
    # Load dataset and encode queries
    qids, questions, answers, _ = load_qa_pairs(args.test_path, args, q_idx)

    if query_encoder is None:
        print(f'Query encoder will be loaded from {args.query_encoder_path}')
        device = 'cuda' if args.cuda else 'cpu'
        query_encoder, tokenizer = load_query_encoder(device, args)
    query_vec = embed_all_query(questions, args, query_encoder, tokenizer)

    # Load MIPS
    if mips is None:
        mips = load_phrase_index(args)

    # Search
    step = args.eval_batch_size
    logger.info(f'Aggergation strategy used: {args.agg_strat}')
    predictions = []
    evidences = []
    titles = []
    scores = []
    se_poss = []
    for q_idx in tqdm(range(0, len(questions), step)):
        result = mips.search(
            query_vec[q_idx:q_idx+step],
            q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
            top_k=args.top_k, max_answer_length=args.max_answer_length,
            aggregate=args.aggregate, agg_strat=args.agg_strat,
        )
        prediction = [[ret['answer'] for ret in out] if len(out) > 0 else [''] for out in result]
        evidence = [[ret['context'] for ret in out] if len(out) > 0 else [''] for out in result]
        title = [[ret['title'] for ret in out] if len(out) > 0 else [['']] for out in result]
        score = [[ret['score'] for ret in out] if len(out) > 0 else [-1e10] for out in result]
        se_pos = [[(ret['start_pos'], ret['end_pos']) for ret in out] if len(out) > 0 else [(0,0)] for out in result]
        predictions += prediction
        evidences += evidence
        titles += title
        scores += score
        se_poss += se_pos

    # logger.info(f"Avg. {sum(mips.num_docs_list)/len(mips.num_docs_list):.2f} number of docs per query")
    eval_fn = evaluate_results if not args.is_kilt else evaluate_results_kilt
    return eval_fn(predictions, qids, questions, answers, args, evidences, scores, titles, se_positions=se_poss)


def evaluate_results(predictions, qids, questions, answers, args, evidences, scores, titles, se_positions=None):
    # Filter if there's candidate
    if args.candidate_path is not None:
        candidates = set()
        with open(args.candidate_path) as f:
            for line in f:
                line = line.strip().lower()
                candidates.add(line)
        logger.info(f'{len(candidates)} candidates are loaded from {args.candidate_path}')
        topk_preds = [list(filter(lambda x: (x in candidates) or (x.lower() in candidates), a)) for a in predictions]
        topk_preds = [a[:args.top_k] if len(a) > 0 else [''] for a in topk_preds]
        predictions = topk_preds[:]
        top1_preds = [a[0] for a in topk_preds]
    else:
        predictions = [a[:args.top_k] if len(a) > 0 else [''] for a in predictions]
        top1_preds = [a[0] for a in predictions]
    no_ans = sum([a == '' for a in top1_preds])
    logger.info(f'no_ans/all: {no_ans}, {len(top1_preds)}')
    logger.info(f'Evaluating {len(top1_preds)} answers')

    # Get em/f1
    f1s, ems = [], []
    for prediction, groundtruth in zip(top1_preds, answers):
        if len(groundtruth)==0:
            f1s.append(0)
            ems.append(0)
            continue
        f1s.append(max([f1_score(prediction, gt)[0] for gt in groundtruth]))
        ems.append(max([exact_match_score(prediction, gt) for gt in groundtruth]))
    final_f1, final_em = np.mean(f1s), np.mean(ems)
    if not args.regex:
        logger.info('EM: %.2f, F1: %.2f'%(final_em * 100, final_f1 * 100))

    # Top 1/k em (or regex em)
    exact_match_topk = 0
    exact_match_top1 = 0
    f1_score_topk = 0
    f1_score_top1 = 0
    redundant_topk = 0
    pred_out = {}
    for i in range(len(predictions)):
        # For debugging
        if i < 3:
            logger.info(f'{i+1}) {questions[i]}')
            logger.info(f'=> groundtruths: {answers[i]}, top 5 prediction: {predictions[i][:5]}')

        match_fn = drqa_regex_match_score if args.regex else drqa_exact_match_score
        em_topk = max([drqa_metric_max_over_ground_truths(
            match_fn, prediction, answers[i]
        ) for prediction in predictions[i][:args.top_k]])
        em_top1 = drqa_metric_max_over_ground_truths(
            match_fn, top1_preds[i], answers[i]
        )
        exact_match_topk += em_topk
        exact_match_top1 += em_top1

        # Compute top-k redundancy (could be ill-defined for regex)
        rd_topk = sum([drqa_metric_max_over_ground_truths(
            match_fn, prediction, [predictions[i][0]]
        ) for prediction in predictions[i][:args.top_k]])
        redundant_topk += rd_topk

        f1_topk = 0
        f1_top1 = 0
        if not args.regex:
            match_fn = lambda x, y: f1_score(x, y)[0]
            f1_topk = max([drqa_metric_max_over_ground_truths(
                match_fn, prediction, answers[i]
            ) for prediction in predictions[i][:args.top_k]])
            f1_top1 = drqa_metric_max_over_ground_truths(
                match_fn, top1_preds[i], answers[i]
            )
            f1_score_topk += f1_topk
            f1_score_top1 += f1_top1

        # Score statistics
        assert len(predictions[i]) <= args.top_k
        pred_out[qids[i]] = {
                'question': questions[i],
                'answer': answers[i], 'prediction': predictions[i], 'score': scores[i], 'title': titles[i],
                'evidence': evidences[i] if evidences is not None else '',
                'em_top1': bool(em_top1), f'em_top{args.top_k}': bool(em_topk),
                'f1_top1': f1_top1, f'f1_top{args.top_k}': f1_topk,
                'se_pos': se_positions[i] if se_positions is not None else (-1, -1),
                'rd_topk': rd_topk,
        }

    total = len(predictions)
    exact_match_top1 = 100.0 * exact_match_top1 / total
    f1_score_top1 = 100.0 * f1_score_top1 / total
    logger.info({'exact_match_top1': exact_match_top1, 'f1_score_top1': f1_score_top1})
    exact_match_topk = 100.0 * exact_match_topk / total
    f1_score_topk = 100.0 * f1_score_topk / total
    logger.info({f'exact_match_top{args.top_k}': exact_match_topk, f'f1_score_top{args.top_k}': f1_score_topk})
    redundant_topk = redundant_topk / total
    logger.info({f'redundancy of top{args.top_k}': redundant_topk})

    # Dump predictions
    if len(args.query_encoder_path) == 0:
        pred_dir = os.path.join(os.environ['SAVE_DIR'], 'pred')
    else:
        pred_dir = os.path.join(args.query_encoder_path, 'pred')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    if args.save_pred:
        pred_path = os.path.join(
            pred_dir, os.path.splitext(os.path.basename(args.test_path))[0] + f'_{total}_top{args.top_k}.pred'
        )
        logger.info(f'Saving prediction file to {pred_path}')
        with open(pred_path, 'w') as f:
            json.dump(pred_out, f)

    return exact_match_top1, f1_score_top1, exact_match_topk, f1_score_topk


def evaluate_results_kilt(predictions, qids, questions, answers, args, evidences, scores, titles, se_positions=None):
    total=len(predictions)

    # load title2id dict and convert predicted titles into wikipedia_ids
    with open(os.path.join(os.environ['DATA_DIR'], args.title2wikiid_path)) as f:
        title2wikiid = json.load(f)
    pred_wikipedia_ids = [[[title2wikiid[t] for t in title_] for title_ in title] for title in titles]

    # dump official predictions
    if len(args.query_encoder_path) == 0:
        pred_dir = os.path.join(os.environ['SAVE_DIR'], 'pred-kilt')
    else:
        pred_dir = os.path.join(args.query_encoder_path, 'pred-kilt')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    pred_official_path = os.path.join(
        pred_dir, f'{args.query_encoder_path.split("/")[-1]}_' +
        os.path.splitext(os.path.basename(args.test_path))[0] + f'_{total}.jsonl'
    )
    official_preds_to_save = []
    for prediction, question, pred_wikipedia_id, qid in zip(predictions, questions, pred_wikipedia_ids, qids):
        outputs = []
        for pred, pred_wid in zip(prediction, pred_wikipedia_id):
            outputs.append({
                'answer': pred,
                'provenance':[{'wikipedia_id':pred_wid_} for pred_wid_ in pred_wid]
            })
            
        official_preds_to_save.append({
            'id': qid,
            'input': question,
            'output': [outputs[0]]
        })

    logger.info(f'Saving official prediction file to {pred_official_path}')
    kilt_store_data(pred_official_path, official_preds_to_save)

    assert '.jsonl' in args.kilt_gold_path, "kilt_gold_path should be .jsonl"
    result = kilt_evaluate(
        gold=args.kilt_gold_path,
        guess=pred_official_path)

    # logging results
    result_to_logging = {
        'accuracy':result['downstream']['accuracy'],
        'f1':result['downstream']['f1'],
        'KILT-accuracy':result['kilt']['KILT-accuracy'],
        'KILT-f1':result['kilt']['KILT-f1'],
        'Rprec':result['retrieval']['Rprec'],
        'recall@5':result['retrieval']['recall@5']
    }

    logger.info(result_to_logging)

    # make custom predictions
    pred_out = {}
    for i in range(len(predictions)):
        # For debugging
        if i < 3:
            logger.info(f'{i+1}) {questions[i]}')
            logger.info(f'=> groundtruths: {answers[i]}, top 5 prediction: {predictions[i][:5]}')

        guess_answer = predictions[i][0]
        gold_candidate_answers = answers[i]
        local_accuracy = 0
        if guess_answer in gold_candidate_answers:
            local_accuracy = 1
        
        pred_out[qids[i]] = {
                'question': questions[i],
                'answer': answers[i], 'prediction': predictions[i], 'score': scores[i], 'title': titles[i],
                'evidence': evidences[i] if evidences is not None else '',
                'em_top1': bool(local_accuracy),
        }

    # dump custom predictions
    pred_path = os.path.join(
        pred_dir, os.path.splitext(os.path.basename(args.test_path))[0] + f'_{total}.pred'
    )
    logger.info(f'Saving custom prediction file to {pred_path}')
    with open(pred_path, 'w') as f:
        json.dump(pred_out, f)

    return result['retrieval']['Rprec'], result['retrieval']['recall@5'], result['kilt']['KILT-accuracy'], result['kilt']['KILT-f1']


def get_hard_negatives(args, mips=None, query_encoder=None, tokenizer=None):
    # Load dataset and encode queries
    if query_encoder is None:
        logger.info(f'Query encoder will be loaded from {args.query_encoder_path}')
        device = 'cuda' if args.cuda else 'cpu'
        query_encoder, tokenizer = load_query_encoder(device, args)
        query2vec = get_query2vec(
            query_encoder=query_encoder, tokenizer=tokenizer, args=args, batch_size=64
        )

    # Load passages
    passages = {}
    logger.info('Loading 21M Wikipedia passages...')
    # with open('/scratch/jinhyuk/paq/psgs_w100.tsv') as f:
    with open('/n/fs/nlp-jl5167/paq/psgs_w100.tsv') as f:
        psg_file = csv.reader(f, delimiter='\t')
        for data_idx, data in tqdm(enumerate(psg_file)):
            if data_idx == 0:
                print('Reading', data)
                continue
            id_, psg, title = data
            passages[psg] = [id_, title]
            # break

    # Load MIPS
    if mips is None:
        mips = load_phrase_index(args)
        logger.info(f'Aggergation strategy used: {args.agg_strat}')
    '''
    '''

    # Example json line
    # {"question":"how many popes have the name gregory",
    #  "subsets":["L1"],
    #  "answer":["Sixteen"],
    #  "answers":[{"passage_id":"2581755","offset":487,"text":"Sixteen","extractor":"L"}],
    #  "passage_score":"-0.06620407104492188"}

    num_split = 8
    partition = 3
    logger.info(f'Partition {partition} of {num_split} splits')
    # with open(args.test_path) as f, open(f'/scratch/jinhyuk/paq/PAQ.metadata.hard0-{partition}.jsonl', 'w') as fw:
    prev_f = open(f'/n/fs/nlp-jl5167/paq/PAQ.metadata.hard0-{partition}-orig.jsonl', 'r')
    with open(args.test_path) as f, open(f'/n/fs/nlp-jl5167/paq/PAQ.metadata.hard0-{partition}.jsonl', 'w') as fw:
        json_list = list(f)
        batch_questions = []
        batch_meta = []

        # check_list = {json.loads(data)['question']: json.loads(data) for data in list(prev_f)[:-1]} # last line broken
        # skip = False
        # if len(check_list) > 0:
        #     logger.info(f'Skip first {len(check_list)} Qs = {num_split * len(check_list)} steps')
        #     skip = True # one time skipper

        for qa_idx, json_str in tqdm(enumerate(json_list), total=len(json_list)):
            if qa_idx % num_split != partition:
                continue
            qa_data = json.loads(json_str) 
            question = qa_data['question']
            question = question[:-1] if question.endswith('?') else question
            
            prev_q = prev_f.readline()
            if len(prev_q) > 0:
                try:
                    prev_data = json.loads(prev_q)
                    assert qa_data['question'] == prev_data['question']
                    skip = True
                except Exception as e:
                    logger.info(e)
                    logger.info('end of prev file')
                    skip = False
            else:
                skip = False

            if skip:
                # json.dump(check_list[qa_data['question']], fw)
                json.dump(prev_data, fw)
                fw.write('\n')
                continue

            answer = qa_data['answer']
            gold_psg_ids = [ans['passage_id'] for ans in qa_data['answers']]

            batch_questions.append(question)
            batch_meta.append(qa_data)

            if (len(batch_questions) == args.eval_batch_size) or (qa_idx + num_split > len(json_list) - 1):
                outs = query2vec(batch_questions)
                start = np.concatenate([out[0] for out in outs], 0)
                end = np.concatenate([out[1] for out in outs], 0)
                query_vec = np.concatenate([start, end], 1)
                
                # Search
                result = mips.search(
                    query_vec,
                    q_texts=batch_questions, nprobe=args.nprobe,
                    top_k=args.top_k, max_answer_length=args.max_answer_length,
                    aggregate=args.aggregate, agg_strat=args.agg_strat,
                )
                context = [[ret['context'] for ret in out] if len(out) > 0 else [''] for out in result]

                # Write
                for ctx_idx, top_ctx in enumerate(context):
                    new_qa_data = copy.deepcopy(batch_meta[ctx_idx])
                    new_qa_data['hard_neg_pids'] = [
                        passages[ctx][0] for ctx in top_ctx
                        if (ctx in passages) and not any(ans in ctx for ans in new_qa_data['answer'])
                    ]
                    new_qa_data['hard_neg_pids'] = list(
                        set(new_qa_data['hard_neg_pids']) - set([ans['passage_id'] for ans in new_qa_data['answers']])
                    )
                    json.dump(new_qa_data, fw)
                    fw.write('\n')

                # Flush
                batch_questions = []
                batch_meta = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # QueryEncoder
    parser.add_argument('--model_type', default='bert', type=str)
    parser.add_argument("--pretrained_name_or_path", default='SpanBERT/spanbert-base-cased', type=str)
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--do_lower_case", default=False, action='store_true')
    parser.add_argument('--max_query_length', default=64, type=int)
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--query_encoder_path", default='', type=str)
    parser.add_argument("--query_port", default='-1', type=str)

    # PhraseIndex
    parser.add_argument('--dump_dir', default='dump')
    parser.add_argument('--phrase_dir', default='phrase')
    parser.add_argument('--index_dir', default='256_flat_SQ4')
    parser.add_argument('--index_name', default='index.faiss')
    parser.add_argument('--idx2id_name', default='idx2id.hdf5')
    parser.add_argument('--index_port', default='-1', type=str)

    # These can be dynamically changed.
    parser.add_argument('--max_answer_length', default=10, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--nprobe', default=256, type=int)
    parser.add_argument('--aggregate', default=False, action='store_true')
    parser.add_argument('--agg_strat', default='opt1', type=str)
    parser.add_argument('--truecase', default=False, action='store_true')
    parser.add_argument("--truecase_path", default='truecase/english_with_questions.dist', type=str)

    # KILT
    parser.add_argument('--is_kilt', default=False, action='store_true')
    parser.add_argument('--kilt_gold_path', default='kilt/trex/trex-dev-kilt.jsonl')
    parser.add_argument('--title2wikiid_path', default='wikidump/title2wikiid.json')
    
    # Serving options
    parser.add_argument('--examples_path', default='examples.txt')

    # Evaluation
    parser.add_argument('--dev_path', default='open-qa/nq-open/dev_preprocessed.json')
    parser.add_argument('--test_path', default='open-qa/nq-open/test_preprocessed.json')
    parser.add_argument('--candidate_path', default=None)
    parser.add_argument('--regex', default=False, action='store_true')
    parser.add_argument('--eval_batch_size', default=64, type=int)

    # Run mode
    parser.add_argument('--run_mode', default='eval')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--save_pred', default=False, action='store_true')
    parser.add_argument('--seed', default=1992, type=int)
    args = parser.parse_args()

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.run_mode == 'eval':
        evaluate(args)

    elif args.run_mode == 'eval_all':
        # Load MIPS & query encoder
        mips = load_phrase_index(args)
        device = 'cuda' if args.cuda else 'cpu'
        query_encoder, tokenizer = load_query_encoder(device, args)

        # Evaluate all test sets
        test_paths = args.test_path.split(',')
        assert all(os.path.exists(path) for path in test_paths)
        logger.info(f"Evaluating {len(test_paths)} datasets: {test_paths}")
        ems = []
        for test_path in test_paths:
            logger.info(f"Evaluating {test_path}")
            new_args = copy.deepcopy(args)
            new_args.test_path = test_path
            if 'trec' in test_path:
                new_args.regex = True
                logger.info('Enable regex for TREC')
            if 'webq' in test_path:
                new_args.candidate_path = os.path.join(os.environ['DATA_DIR'], 'open-qa/webq/freebase-entities.txt')
                logger.info('Enable candidates for WebQuestions')
            em, _, _, _ = evaluate(new_args, mips, query_encoder, tokenizer)
            ems.append(f'{em:.1f}')
        logger.info(f"Results of {args.query_encoder_path}")
        print(f'Top1 EMs: {" ".join(ems)}')
    
    elif args.run_mode == 'get_hard_neg':
        get_hard_negatives(args)

    else:
        raise NotImplementedError
