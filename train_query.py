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

from time import time
from tqdm import tqdm
from densephrases.utils.squad_utils import get_question_dataloader
from densephrases.utils.open_utils import load_query_encoder, load_phrase_index, get_query2vec, load_qa_pairs
from densephrases.utils.eval_utils import drqa_exact_match_score, drqa_regex_match_score, \
                                          drqa_metric_max_over_ground_truths
from eval_phrase_retrieval import evaluate

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
server = None


def train_query_encoder(args, mips=None):
    # Freeze one for MIPS
    device = 'cuda' if args.cuda else 'cpu'
    logger.info("Loading pretrained encoder: this one is for MIPS (fixed)")
    pretrained_encoder, tokenizer = load_query_encoder(device, args)

    # Train another
    logger.info("Loading target encoder: this one is for training")
    target_encoder, _= load_query_encoder(device, args)

    # MIPS
    if mips is None:
        mips = load_phrase_index(args)

    # Optimizer setting
    def is_train_param(name):
        if name.endswith(".embeddings.word_embeddings.weight"):
            logger.info(f'freezing {name}')
            return False
        return True
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
            "params": [
                p for n, p in target_encoder.named_parameters() \
                    if not any(nd in n for nd in no_decay) and is_train_param(n)
            ],
            "weight_decay": 0.01,
        }, {
            "params": [
                p for n, p in target_encoder.named_parameters() \
                    if any(nd in n for nd in no_decay) and is_train_param(n)
            ],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    step_per_epoch = math.ceil(len(load_qa_pairs(args.train_path, args)[1]) / args.per_gpu_train_batch_size)
    t_total = int(step_per_epoch // args.gradient_accumulation_steps * args.num_train_epochs)
    logger.info(f"Train for {t_total} iterations")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
     )
    eval_steps = math.ceil(len(load_qa_pairs(args.dev_path, args)[1]) / args.eval_batch_size)
    logger.info(f"Test takes {eval_steps} iterations")

    # Train arguments
    args.per_gpu_train_batch_size = int(args.per_gpu_train_batch_size / args.gradient_accumulation_steps)
    best_acc = -1000.0
    for ep_idx in range(int(args.num_train_epochs)):

        # Training
        total_loss = 0.0
        total_accs = []
        total_accs_k = []

        # Load training dataset
        q_ids, questions, answers, titles = load_qa_pairs(args.train_path, args, shuffle=True)
        pbar = tqdm(get_top_phrases(
            mips, q_ids, questions, answers, titles, pretrained_encoder, tokenizer,
            args.per_gpu_train_batch_size, args)
        )

        for step_idx, (q_ids, questions, answers, titles, outs) in enumerate(pbar):
            train_dataloader, _, _ = get_question_dataloader(
                questions, tokenizer, args.max_query_length, batch_size=args.per_gpu_train_batch_size
            )
            svs, evs, tgts, p_tgts = get_phrase_vecs(mips, q_ids, questions, answers, titles, outs, args)

            target_encoder.train()
            svs_t = torch.Tensor(svs).to(device)
            evs_t = torch.Tensor(evs).to(device)
            tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in tgts]
            if p_tgts is not None:
                p_tgts_t = [torch.Tensor([sp_ for sp_ in sp if sp_ is not None]).to(device) for sp in p_tgts]

            # Train query encoder
            assert len(train_dataloader) == 1
            for batch in train_dataloader:
                batch = tuple(t.to(device) for t in batch)
                loss, accs = target_encoder.train_query(
                    input_ids_=batch[0], attention_mask_=batch[1], token_type_ids_=batch[2],
                    start_vecs=svs_t,
                    end_vecs=evs_t,
                    targets=tgts_t,
                    p_targets=p_tgts_t if p_tgts is not None else None,
                )

                # Optimize, get acc and report
                if loss is not None:
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    total_loss += loss.mean().item()
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(target_encoder.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    target_encoder.zero_grad()

                    pbar.set_description(
                        f"Ep {ep_idx+1} Tr loss: {loss.mean().item():.2f}, acc: {sum(accs)/len(accs):.3f}"
                    )

                if accs is not None:
                    total_accs += accs
                    total_accs_k += [len(tgt) > 0 for tgt in tgts_t]
                else:
                    total_accs += [0.0]*len(tgts_t)
                    total_accs_k += [0.0]*len(tgts_t)

        step_idx += 1
        logger.info(
            f"Avg train loss ({step_idx} iterations): {total_loss/step_idx:.2f} | train " +
            f"acc@1: {sum(total_accs)/len(total_accs):.3f} | acc@{args.top_k}: {sum(total_accs_k)/len(total_accs_k):.3f}"
        )

        # Evaluation
        new_args = copy.deepcopy(args)
        new_args.top_k = 10
        new_args.save_pred = False
        new_args.test_path = args.dev_path
        dev_em, dev_f1, dev_emk, dev_f1k = evaluate(new_args, mips, target_encoder, tokenizer)
        logger.info(f"Develoment set acc@1: {dev_em:.3f}, f1@1: {dev_f1:.3f}")

        # Save best model
        if dev_em > best_acc:
            best_acc = dev_em
            save_path = args.output_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            target_encoder.save_pretrained(save_path)
            logger.info(f"Saved best model with acc {best_acc:.3f} into {save_path}")

        if (ep_idx + 1) % 1 == 0:
            logger.info('Updating pretrained encoder')
            pretrained_encoder = copy.deepcopy(target_encoder)

    print()
    logger.info(f"Best model has acc {best_acc:.3f} saved as {save_path}")


def get_top_phrases(mips, q_ids, questions, answers, titles, query_encoder, tokenizer, batch_size, args):
    # Search
    step = batch_size
    phrase_idxs = []
    search_fn = mips.search
    query2vec = get_query2vec(
        query_encoder=query_encoder, tokenizer=tokenizer, args=args, batch_size=batch_size
    )
    for q_idx in tqdm(range(0, len(questions), step)):
        outs = query2vec(questions[q_idx:q_idx+step])
        start = np.concatenate([out[0] for out in outs], 0)
        end = np.concatenate([out[1] for out in outs], 0)
        query_vec = np.concatenate([start, end], 1)

        outs = search_fn(
            query_vec,
            q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
            top_k=args.top_k, return_idxs=True,
            max_answer_length=args.max_answer_length, aggregate=args.aggregate, agg_strat=args.agg_strat,
        )
        yield (
            q_ids[q_idx:q_idx+step], questions[q_idx:q_idx+step], answers[q_idx:q_idx+step],
            titles[q_idx:q_idx+step], outs
        )


def get_phrase_vecs(mips, q_ids, questions, answers, titles, outs, args, label_strategy='gold'):
    assert mips is not None

    # Get phrase and vectors
    phrase_idxs = [[(out_['doc_idx'], out_['start_idx'], out_['end_idx'], out_['answer'],
        out_['start_vec'], out_['end_vec'], out_['context'], out_['title']) for out_ in out]
        for out in outs
    ]
    batch_size = len(answers)
    default_doc = phrase_idxs[0][0][0]
    for b_idx, phrase_idx in enumerate(phrase_idxs):
        while len(phrase_idxs[b_idx]) < args.top_k*2: # two separate top-k from start/end
            phrase_idxs[b_idx].append((-1, 0, 0, '', np.zeros((768)), np.zeros((768)), '', ''))
        phrase_idxs[b_idx] = phrase_idxs[b_idx][:args.top_k*2]
    flat_phrase_idxs = [phrase for phrase_idx in phrase_idxs for phrase in phrase_idx]
    doc_idxs = [int(phrase_idx_[0]) for phrase_idx_ in flat_phrase_idxs]
    start_idxs = [int(phrase_idx_[1]) for phrase_idx_ in flat_phrase_idxs]
    end_idxs = [int(phrase_idx_[2]) for phrase_idx_ in flat_phrase_idxs]
    phrases = [phrase_idx_[3] for phrase_idx_ in flat_phrase_idxs]
    start_vecs = [phrase_idx_[4] for phrase_idx_ in flat_phrase_idxs]
    end_vecs = [phrase_idx_[5] for phrase_idx_ in flat_phrase_idxs]
    
    # stack vectors
    start_vecs = np.stack(
        [start_vec for start_vec, start_idx in zip(start_vecs, start_idxs)]
    )
    end_vecs = np.stack(
        [end_vec for end_vec, end_idx in zip(end_vecs, end_idxs)]
    )

    zero_mask = np.array([[1] if len(phrase) > 0 else [0] for phrase in phrases])
    start_vecs = start_vecs * zero_mask
    end_vecs = end_vecs * zero_mask

    # Reshape
    start_vecs = np.reshape(start_vecs, (batch_size, args.top_k*2, -1))
    end_vecs = np.reshape(end_vecs, (batch_size, args.top_k*2, -1))

    # Get cross-encoder results
    '''
    result = server.batch_query(
        [q for question in questions for q in [question]*args.top_k*2],
        [phrase[6] for phrase_idx in phrase_idxs for phrase in phrase_idx],
    )
    ce_answers = [val['answer'] for val in list(result['ret'].values())]
    ce_scores = [val['score'] for val in list(result['ret'].values())]
    top_idxs = np.array(ce_scores).argsort()[-10:][::-1]
    '''

    # Find targets based on exact string match
    match_fns = [
        drqa_regex_match_score if args.regex or ('trec' in q_id.lower()) else drqa_exact_match_score for q_id in q_ids
    ]
    no_phrase_tasks = ['fever', 'hotpot', 'eli5', 'wow']
    train_phrase = [
        False
        # False if any(task in q_id.lower() for task in no_phrase_tasks) or len(ans[0].split()) > 10 else True
        for q_id, ans in zip(q_ids, answers)
    ]

    if label_strategy == 'gold':
        targets = [
            [drqa_metric_max_over_ground_truths(match_fn, phrase[3], answer) and train_p for phrase in phrase_idx]
            for b_idx, (phrase_idx, answer, match_fn, train_p, question) in enumerate(zip(phrase_idxs, answers, match_fns, train_phrase, questions))
        ]
    elif label_strategy == 'pseudo':
        targets = [
            [(phrase[7][0].lower() in question.lower()) for phrase in phrase_idx]
            for b_idx, (phrase_idx, answer, match_fn, train_p, question) in enumerate(zip(phrase_idxs, answers, match_fns, train_phrase, questions))
        ]
    else:
        raise NotImplementedError('invalid strategy')
    targets = [[ii if val else None for ii, val in enumerate(target)] for target in targets]

    # Passage (or article) level target
    p_targets = [
        [any(phrase[7][0].lower() == tit.lower() for tit in title) for phrase in phrase_idx] # article
        # [any(ans.lower() in phrase[6].lower() for ans in answer) for phrase in phrase_idx] # passage
        for phrase_idx, answer, title, match_fn in zip(phrase_idxs, answers, titles, match_fns)
    ]
    p_targets = [[ii if val else None for ii, val in enumerate(target)] for target in p_targets]

    # For debug
    def print_ds(kk):
        print('QA pair:', questions[kk], answers[kk])
        print('DS target idxs:', [t for t in targets[kk] if t is not None])
        print('DS target texts:', target_texts[kk])
        # print('all predictions:', all_texts[kk][:10])
        for kk_ in targets[kk]:
            if kk_ is not None:
                print('\nEvidence snippet:', texts[kk][kk_][0])
                print('Evidence doc:', texts[kk][kk_][1])
    # import pdb; pdb.set_trace()

    return start_vecs, end_vecs, targets, p_targets


def test_query(args, mips, pretrained_encoder, tokenizer, target_encoder, data, q_idx):

    # Optimizer setting
    def is_train_param(name):
        if name.endswith(".embeddings.word_embeddings.weight"):
            # logger.info(f'freezing {name}')
            return False
        return True
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
            "params": [
                p for n, p in target_encoder.named_parameters() \
                    if not any(nd in n for nd in no_decay) and is_train_param(n)
            ],
            "weight_decay": 0.01,
        }, {
            "params": [
                p for n, p in target_encoder.named_parameters() \
                    if any(nd in n for nd in no_decay) and is_train_param(n)
            ],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    step_per_epoch = 1
    t_total = int(step_per_epoch // args.gradient_accumulation_steps * args.num_train_epochs)
    logger.info(f"Train for {t_total} iterations")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
     )

    # Train arguments
    args.per_gpu_train_batch_size = int(args.per_gpu_train_batch_size / args.gradient_accumulation_steps)
    best_acc = -1000.0
    for ep_idx in range(int(args.num_train_epochs)):

        # Evaluation
        new_args = copy.deepcopy(args)
        new_args.save_pred = False
        new_args.aggregate = True
        dev_em, dev_f1, dev_emk, dev_f1k = evaluate(new_args, mips, target_encoder, tokenizer, q_idx=q_idx)
        logger.info(f"[Epoch {ep_idx+1}] test acc@1: {dev_em:.3f}, f1@1: {dev_f1:.3f}")

        # Training
        total_loss = 0.0
        total_accs = []
        total_accs_k = []

        # Load training dataset
        # q_ids, questions, answers, titles = load_qa_pairs(args.train_path, args, shuffle=True)
        q_ids, questions, answers, titles = data
        pbar = tqdm(get_top_phrases(
            mips, q_ids, questions, answers, titles, pretrained_encoder, tokenizer,
            args.per_gpu_train_batch_size, args)
        )

        for step_idx, (q_ids, questions, answers, titles, outs) in enumerate(pbar):
            train_dataloader, _, _ = get_question_dataloader(
                questions, tokenizer, args.max_query_length, batch_size=args.per_gpu_train_batch_size
            )
            svs, evs, tgts, p_tgts = get_phrase_vecs(mips, q_ids, questions, answers, titles, outs, args, label_strategy='pseudo')

            target_encoder.train()
            svs_t = torch.Tensor(svs).to(device)
            evs_t = torch.Tensor(evs).to(device)
            tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in tgts]
            if p_tgts is not None:
                p_tgts_t = [torch.Tensor([sp_ for sp_ in sp if sp_ is not None]).to(device) for sp in p_tgts]

            # Train query encoder
            assert len(train_dataloader) == 1
            for batch in train_dataloader:
                batch = tuple(t.to(device) for t in batch)
                loss, accs = target_encoder.train_query(
                    input_ids_=batch[0], attention_mask_=batch[1], token_type_ids_=batch[2],
                    start_vecs=svs_t,
                    end_vecs=evs_t,
                    targets=tgts_t,
                    p_targets=p_tgts_t if p_tgts is not None else None,
                )

                # Optimize, get acc and report
                if loss is not None:
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    total_loss += loss.mean().item()
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(target_encoder.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    target_encoder.zero_grad()

                    pbar.set_description(
                        f"Ep {ep_idx+1} Tr loss: {loss.mean().item():.2f}, acc: {sum(accs)/len(accs):.3f}"
                    )

                if accs is not None:
                    total_accs += accs
                    total_accs_k += [len(tgt) > 0 for tgt in tgts_t]
                else:
                    total_accs += [0.0]*len(tgts_t)
                    total_accs_k += [0.0]*len(tgts_t)

        step_idx += 1
        logger.info(
            f"Avg train loss ({step_idx} iterations): {total_loss/step_idx:.2f} | train " +
            f"acc@1: {sum(total_accs)/len(total_accs):.3f} | acc@{args.top_k}: {sum(total_accs_k)/len(total_accs_k):.3f}"
        )

        if (0. in tgts_t[0]) or (len(tgts_t[0]) == 0):
            return dev_em, dev_f1, dev_emk, dev_f1k

        # Save best model
        if dev_f1 > best_acc:
            best_acc = dev_f1
            # logger.info(f"Best model with acc {best_acc:.3f} into {save_path}")

        if (ep_idx + 1) % 1 == 0:
            # logger.info('Updating pretrained encoder')
            pretrained_encoder = copy.deepcopy(target_encoder)

    # Final evaluation
    new_args = copy.deepcopy(args)
    new_args.save_pred = False
    new_args.aggregate = True
    dev_em, dev_f1, dev_emk, dev_f1k = evaluate(new_args, mips, target_encoder, tokenizer, q_idx=q_idx)
    logger.info(f"[Final] test acc@1: {dev_em:.3f}, f1@1: {dev_f1:.3f}")

    return dev_em, dev_f1, dev_emk, dev_f1k


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

    # Training query encoder
    parser.add_argument('--train_path', default=None)
    parser.add_argument('--per_gpu_train_batch_size', default=48, type=int)
    parser.add_argument('--num_train_epochs', default=10, type=float)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--warmup_steps", default=0.1, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--use_inbatch_neg", action="store_true", help="Whether to run with inb-neg.")
    parser.add_argument("--fp16", action="store_true", help="For fp16")
    parser.add_argument('--output_dir', default=None, type=str)

    # Evaluation
    parser.add_argument('--dev_path', default='open-qa/nq-open/dev_preprocessed.json')
    parser.add_argument('--test_path', default='open-qa/nq-open/test_preprocessed.json')
    parser.add_argument('--candidate_path', default=None)
    parser.add_argument('--regex', default=False, action='store_true')
    parser.add_argument('--eval_batch_size', default=10, type=int)
    parser.add_argument('--filter_threshold', default=None, type=float)

    # Run mode
    parser.add_argument('--run_mode', default='train_query')
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

    from run_demo import DensePhrasesInterface
    args.index_port = '1111'
    args.base_ip = 'http://127.0.0.1'
    server = DensePhrasesInterface(args)

    if args.run_mode == 'train_query':
        # Train
        mips = load_phrase_index(args)
        train_query_encoder(args, mips)

        # Eval
        args.query_encoder_path = args.output_dir
        logger.info(f"Evaluating {args.query_encoder_path}")
        args.top_k = 10
        evaluate(args, mips)

    elif args.run_mode == 'test_query':
        logging.getLogger("eval_phrase_retrieval").setLevel(logging.DEBUG)
        logging.getLogger("densephrases.utils.open_utils").setLevel(logging.DEBUG)

        # Train
        mips = load_phrase_index(args)

        # Query encoder
        device = 'cuda' if args.cuda else 'cpu'
        logger.info("Loading pretrained encoder: this one is for MIPS (fixed)")
        pretrained_encoder, tokenizer = load_query_encoder(device, args)

        # Load test questions
        q_ids, questions, answers, titles = load_qa_pairs(args.test_path, args, shuffle=False)

        ems = []
        f1s = []
        emks = []
        f1ks = []
        for q_idx, (q_id, question, answer, title) in tqdm(enumerate(zip(q_ids, questions, answers, titles)), total=len(questions)):
            em, f1, emk, f1k = test_query(
                args, mips, copy.deepcopy(pretrained_encoder), tokenizer, copy.deepcopy(pretrained_encoder),
                data=[[q_id], [question], [answer], [title]], q_idx=q_idx
            )
            ems.append(em)
            f1s.append(f1)
            emks.append(emk)
            f1ks.append(f1k)

        logger.info(f"Test-time QSFT on {len(ems)} questions")
        logger.info(f"Acc={sum(ems)/len(ems):.2f} | F1={sum(f1s)/len(f1s):.2f}")
        logger.info(f"Acc@{args.top_k}={sum(emks)/len(emks):.2f} | F1@{args.top_k}={sum(f1ks)/len(f1ks):.2f}")

    else:
        raise NotImplementedError
