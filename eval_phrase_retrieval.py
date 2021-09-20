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
import subprocess

from time import time
from tqdm import tqdm

from densephrases.utils.eval_utils import normalize_answer, f1_score, exact_match_score, drqa_exact_match_score, \
        drqa_regex_match_score, drqa_metric_max_over_ground_truths, drqa_normalize
from densephrases.utils.single_utils import load_encoder
from densephrases.utils.open_utils import load_phrase_index, get_query2vec, load_qa_pairs
from densephrases.utils.kilt.eval import evaluate as kilt_evaluate
from densephrases.utils.kilt.kilt_utils import store_data as kilt_store_data
from densephrases import Options


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
        logger.info(f'Query encoder will be loaded from {args.load_dir}')
        device = 'cuda' if args.cuda else 'cpu'
        query_encoder, tokenizer, _ = load_encoder(device, args)
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
            aggregate=args.aggregate, agg_strat=args.agg_strat, return_sent=args.return_sent
        )
        prediction = [[ret['answer'] for ret in out][:args.top_k] if len(out) > 0 else [''] for out in result]
        evidence = [[ret['context'] for ret in out][:args.top_k] if len(out) > 0 else [''] for out in result]
        title = [[ret['title'] for ret in out][:args.top_k] if len(out) > 0 else [['']] for out in result]
        score = [[ret['score'] for ret in out][:args.top_k] if len(out) > 0 else [-1e10] for out in result]
        se_pos = [[(ret['start_pos'], ret['end_pos']) for ret in out][:args.top_k] if len(out) > 0 else [(0,0)] for out in result]
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
    if len(args.load_dir) == 0:
        pred_dir = os.path.join(os.environ['SAVE_DIR'], 'pred')
    else:
        pred_dir = os.path.join(args.load_dir, 'pred')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    if args.save_pred:
        pred_path = os.path.join(
            pred_dir, os.path.splitext(os.path.basename(args.test_path))[0] + f'_{total}_top{args.top_k}.pred'
        )
        logger.info(f'Saving prediction file to {pred_path}')
        with open(pred_path, 'w') as f:
            json.dump(pred_out, f)

    # Evaluate passage retrieval
    if args.eval_psg:
        evaluate_results_psg(pred_path, args)

    return exact_match_top1, f1_score_top1, exact_match_topk, f1_score_topk


def evaluate_results_kilt(predictions, qids, questions, answers, args, evidences, scores, titles, se_positions=None):
    total=len(predictions)

    # load title2id dict and convert predicted titles into wikipedia_ids
    with open(args.title2wikiid_path) as f:
        title2wikiid = json.load(f)
    pred_wikipedia_ids = [[[title2wikiid[t] for t in title_] for title_ in title] for title in titles]

    # dump official predictions
    if len(args.load_dir) == 0:
        pred_dir = os.path.join(os.environ['SAVE_DIR'], 'pred-kilt')
    else:
        pred_dir = os.path.join(args.load_dir, 'pred-kilt')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    pred_official_path = os.path.join(
        pred_dir, f'{args.load_dir.split("/")[-1]}_' +
        os.path.splitext(os.path.basename(args.test_path))[0] + f'_{total}.jsonl'
    )
    official_preds_to_save = []
    for prediction, title, question, pred_wikipedia_id, qid in zip(predictions, titles, questions, pred_wikipedia_ids, qids):
        if ("wned" in pred_official_path or
            "cweb" in pred_official_path or
            "aidayago2" in pred_official_path):
            answer = title[0][0]
        else:
            answer = prediction[0].strip(string.punctuation)

        output = {
            'answer': answer,
            'provenance': [{'wikipedia_id': pred_wid_} for pred_wid in pred_wikipedia_id for pred_wid_ in pred_wid]
        }
        official_preds_to_save.append({
            'id': qid,
            'input': question,
            'output': [output]
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


def evaluate_results_psg(pred_path, args):
    # Read prediction
    my_pred = json.load(open(pred_path))

    my_target = []
    avg_len = []
    for qid, pred in tqdm(enumerate(my_pred.values())):
        my_dict = {"id": str(qid), "question": None, "answers": [], "ctxs": []}

        # truncate
        pred = {key: val[:args.psg_top_k] if key in ['evidence', 'title', 'se_pos', 'prediction'] else val for key, val in pred.items()}

        # TODO: need to add id for predictions.pred
        my_dict["question"] = pred["question"]
        my_dict["answers"] = pred["answer"]
        pred["title"] = [titles[0] for titles in pred["title"]]

        assert len(set(pred["evidence"])) == len(pred["evidence"]) == len(pred["title"]), "Should use opt2 for aggregation"
        # assert all(pr in evd for pr, evd in zip(pred["prediction"], pred["evidence"]))  # prediction included TODO: fails when return_sent=True

        # Pad up to top-k
        if not(len(pred["prediction"]) == len(pred["evidence"]) == len(pred["title"]) == args.psg_top_k):
            assert len(pred["prediction"]) == len(pred["evidence"]) == len(pred["title"]) < args.psg_top_k, \
                (len(pred["prediction"]), len(pred["evidence"]), len(pred["title"]))
            # logger.info(len(pred["prediction"]), len(pred["evidence"]), len(pred["title"]))

            pred["evidence"] += [pred["evidence"][-1]] * (args.psg_top_k - len(pred["prediction"]))
            pred["title"] += [pred["title"][-1]] * (args.psg_top_k - len(pred["prediction"]))
            pred["se_pos"] += [pred["se_pos"][-1]] * (args.psg_top_k - len(pred["prediction"]))
            pred["prediction"] += [pred["prediction"][-1]] * (args.psg_top_k - len(pred["prediction"]))
            assert len(pred["prediction"]) == len(pred["evidence"]) == len(pred["title"]) == args.psg_top_k

        # Used for markers
        START = '<p_start>'
        END = '<p_end>'
        se_idxs = [[se_pos[0], max(se_pos[0], se_pos[1])] for se_pos in pred["se_pos"]]

        # cut based on max psg len
        my_dict["ctxs"] = [
            {"title": title, "text": ' '.join(evd.split()[:args.max_psg_len])}
            for evd, title in zip(pred["evidence"], pred["title"])
        ]

        # Add markers for predicted phrases
        if args.mark_phrase:
            my_dict["ctxs"] = [
                {"title": ctx["title"], "text": ctx["text"][:se[0]] + f"{START} " + ctx["text"][se[0]:se[1]] + f" {END}" + ctx["text"][se[1]:]}
                for ctx, se in zip(my_dict["ctxs"], se_idxs)
            ]

        my_target.append(my_dict)
        avg_len += [len(ctx['text'].split()) for ctx in my_dict["ctxs"]]
        assert len(my_dict["ctxs"]) == args.psg_top_k
        assert all(len(ctx['text'].split()) <= args.max_psg_len for ctx in my_dict["ctxs"])

    logger.info(f"avg psg len={sum(avg_len)/len(avg_len):.2f} for {len(my_pred)} preds")

    out_file = os.path.join(
        os.environ['SAVE_DIR'], os.path.basename(args.load_dir), 'pred',
        os.path.splitext(os.path.basename(pred_path))[0] + 
        f'_{"sent" if args.return_sent else "psg"}-top{args.psg_top_k}{"_mark" if args.mark_phrase else ""}.json'
    )
    logger.info(f"dump to {out_file}")
    json.dump(my_target, open(out_file, 'w'), indent=4)

    # Call subprocess for evaluation
    command = f'python scripts/postprocess/recall.py --k_values 1,5,20,100 --results_file {out_file} --ans_fn string'
    subprocess.run(command.split(' '))


if __name__ == '__main__':
    # See options in densephrases.options
    options = Options()
    options.add_model_options()
    options.add_index_options()
    options.add_retrieval_options()
    options.add_data_options()
    args = options.parse()

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
        query_encoder, tokenizer, _ = load_encoder(device, args)

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
        logger.info(f"Results of {args.load_dir}")
        logger.info(f'Top1 EMs: {" ".join(ems)}')
    
    else:
        raise NotImplementedError
