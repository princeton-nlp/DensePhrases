""" Very heavily inspired by the official evaluation script for SQuAD version 2.0 which was
modified by XLNet authors to update `find_best_threshold` scripts for SQuAD V2.0

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""


import collections
import json
import logging
import math
import re
import string
import tqdm
import h5py
import numpy as np
import torch

from multiprocessing import Queue, Process
from threading import Thread
from time import time
from tqdm import tqdm
from transformers.tokenization_bert import BasicTokenizer

from .squad_utils import QuestionResult, SquadResult
from .squad_metrics import get_final_text_


logger = logging.getLogger(__name__)
# For debugging
quant_stat = {}
b_quant_stat = {}

id2example = None


def get_metadata(features, results, max_answer_length, do_lower_case, tokenizer, verbose_logging, has_title):
    global id2example

    # Get rid of titles + save start only (as start and end are shared)
    roberta_add = 1 if "roberta" in str(type(tokenizer)) else 0
    toffs = [(f.input_ids.index(tokenizer.sep_token_id))*int(has_title) + roberta_add for f in features]

    # Filter reps
    fs = np.concatenate(
        [result.sft_logits[to+1:len(feature.tokens) - 1] for feature, result, to in zip(features, results, toffs)], axis=0
    )
    fe = np.concatenate(
        [result.eft_logits[to+1:len(feature.tokens) - 1] for feature, result, to in zip(features, results, toffs)], axis=0
    )

    if max_answer_length is None:
        example = id2example[features[-1].unique_id]
        metadata = {
            'did': example.doc_idx, 'title': example.title,
            'filter_start': fs, 'filter_end': fe
        }
        return metadata

    # start vectors
    start = np.concatenate(
        [result.start_vecs[to+1:len(feature.tokens) - 1] for feature, result, to in zip(features, results, toffs)],
        axis=0
    )

    len_per_para = [len(f.input_ids[to+1:len(f.tokens)-1]) for to, f in zip(toffs, features)]
    curr_size = 0

    # Start2end map
    start2end = -1 * np.ones([np.shape(start)[0], max_answer_length], dtype=np.int32)
    idx = 0
    for feature, result, to in zip(features, results, toffs):
        for i in range(to+1, len(feature.tokens) - 1):
            for j in range(i, min(i + max_answer_length, len(feature.tokens) - 1)):
                start2end[idx, j - i] = idx + j - i
            idx += 1

    word2char_start = np.zeros([start.shape[0]], dtype=np.int32)
    word2char_end = np.zeros([start.shape[0]], dtype=np.int32)

    # Orig map
    sep = ' [PAR] '
    full_text = ""
    prev_example = None
    word_pos = 0
    for feature, to in zip(features, toffs):
        example = id2example[feature.unique_id]
        if prev_example is not None and feature.span_idx == 0:
            full_text = full_text + ' '.join(prev_example.doc_tokens) + sep

        for i in range(to+1, len(feature.tokens) - 1):
            _, start_pos, _ = get_final_text_(example, feature, i, min(len(feature.tokens) - 2, i + 1), do_lower_case,
                                              tokenizer, verbose_logging)
            _, _, end_pos = get_final_text_(example, feature, max(to+1, i - 1), i, do_lower_case,
                                            tokenizer, verbose_logging)
            start_pos += len(full_text)
            end_pos += len(full_text)
            word2char_start[word_pos] = start_pos
            word2char_end[word_pos] = end_pos
            word_pos += 1
        prev_example = example
    full_text = full_text + ' '.join(prev_example.doc_tokens)

    metadata = {
        'did': prev_example.doc_idx, 'context': full_text, 'title': prev_example.title,
        'start': start, 'start2end': start2end,
        'word2char_start': word2char_start, 'word2char_end': word2char_end,
        'filter_start': fs, 'filter_end': fe, 'len_per_para': len_per_para
    }

    return metadata


def filter_metadata(metadata, threshold):
    start_idxs, = np.where(metadata['filter_start'] > threshold)
    end_idxs, = np.where(metadata['filter_end'] > threshold)
    all_idxs = np.array(sorted(list(set(np.concatenate([start_idxs, end_idxs])))))
    end_long2short = {long: short for short, long in enumerate(all_idxs) if long in end_idxs} # fixed for end_idx
    # print(all_idxs)
    # print(end_long2short)

    if len(all_idxs) == 0:
        all_idxs = np.where(metadata['filter_start'] > -999999)[0][:1] # just get all
        end_long2short = {long: short for short, long in enumerate(all_idxs)}
        print('all idxs were filtered, so use only one vector for this:', len(all_idxs))
    metadata['start'] = metadata['start'][all_idxs] # union of start/end
    metadata['f2o_start'] = all_idxs
    metadata['start2end'] = metadata['start2end'][all_idxs]
    # print(metadata['start2end'])
    for i, each in enumerate(metadata['start2end']):
        for j, long in enumerate(each.tolist()):
            metadata['start2end'][i, j] = end_long2short[long] if long in end_long2short else -1
    # print(metadata['start2end'])

    return metadata


def float_to_int8(num, offset, factor):
    out = (num - offset) * factor
    out = out.clip(-128, 127)
    out = np.round(out).astype(np.int8)
    return out


def int8_to_float(num, offset, factor):
    return num.astype(np.float32) / factor + offset


def float_to_int4(num, offset=-3.5, factor=2.3):
    out = (num - offset) * factor
    out = out.clip(0, 16)
    out = np.round(out).astype(np.uint8)

    hd = out.shape[1] // 2
    merged = out[:,:hd] * 16 + out[:,hd:]
    merged = merged.clip(0, 255)
    return merged


def int4_to_float(num, offset=-3.5, factor=2.3):
    unmerged = np.concatenate((num // 16, num % 16), axis=1)
    return unmerged.astype(np.float32) / factor + offset


def compress_metadata(metadata, dense_offset, dense_scale):
    for key in ['start']:
        if key in metadata:
            '''
            if key == 'start':
                for meta in metadata[key]:
                    for number in meta:
                        num_str = "%.1f" % number
                        if float(num_str) not in b_quant_stat:
                            b_quant_stat[float(num_str)] = 0
                        b_quant_stat[float(num_str)] += 1
            '''
            metadata[key] = float_to_int8(metadata[key], dense_offset, dense_scale)
            # metadata[key] = float_to_int4(metadata[key])
            '''
            if key == 'start':
                for meta in metadata[key]:
                    for number in meta:
                        num_str = "%d" % number
                        if int(num_str) not in quant_stat:
                            quant_stat[int(num_str)] = 0
                        quant_stat[int(num_str)] += 1
            '''
    return metadata


def pool_func(item):
    metadata_ = get_metadata(*item[:-1])
    if 'start' in metadata_:
        metadata_ = filter_metadata(metadata_, item[-1])
    return metadata_


def write_phrases(all_examples, all_features, all_results, max_answer_length, do_lower_case, tokenizer, hdf5_path,
    filter_threshold, verbose_logging, dense_offset=None, dense_scale=None, has_title=False):

    assert len(all_examples) > 0

    id2feature = {feature.unique_id: feature for feature in all_features}
    id2example_ = {id_: all_examples[id2feature[id_].example_index] for id_ in id2feature}

    def add(inqueue_, outqueue_):
        for item in iter(inqueue_.get, None):
            # start_time = time()
            args = list(item[:2]) + [
                max_answer_length, do_lower_case, tokenizer, verbose_logging, has_title, filter_threshold
            ]
            out = pool_func(args)
            # print(f'in {time() - start_time:.1f} sec, {inqueue_.qsize()}')
            outqueue_.put(out)

        outqueue_.put(None)

    def write(outqueue_):
        with h5py.File(hdf5_path, 'a') as f:
            while True:
                metadata = outqueue_.get()
                if metadata:
                    # start_time = time()
                    did = str(metadata['did'])
                    if did in f:
                        logger.info('%s exists; replacing' % did)
                        del f[did]
                        # logger.info('%s exists; skipping' % did)
                        # continue
                    dg = f.create_group(did)

                    dg.attrs['context'] = metadata['context']
                    dg.attrs['title'] = metadata['title']
                    if dense_offset is not None:
                        metadata = compress_metadata(metadata, dense_offset, dense_scale)
                        dg.attrs['offset'] = dense_offset
                        dg.attrs['scale'] = dense_scale
                    dg.create_dataset('start', data=metadata['start'])
                    dg.create_dataset('len_per_para', data=metadata['len_per_para'])
                    dg.create_dataset('start2end', data=metadata['start2end'])
                    dg.create_dataset('word2char_start', data=metadata['word2char_start'])
                    dg.create_dataset('word2char_end', data=metadata['word2char_end'])
                    dg.create_dataset('f2o_start', data=metadata['f2o_start'])
                    # print(f'out {time() - start_time:.1f} sec, {outqueue_.qsize()} ')
                else:
                    break

    features = []
    results = []
    inqueue = Queue(maxsize=50)
    outqueue = Queue(maxsize=50)
    NUM_THREAD = 10
    in_p_list = [Process(target=add, args=(inqueue, outqueue)) for _ in range(NUM_THREAD)]
    out_p_list = [Thread(target=write, args=(outqueue,)) for _ in range(NUM_THREAD)]
    global id2example
    id2example = id2example_
    for in_p in in_p_list:
        in_p.start()
    for out_p in out_p_list:
        out_p.start()

    start_time = time()
    for count, result in enumerate(tqdm(all_results, total=len(all_features))):
        example = id2example[result.unique_id]
        feature = id2feature[result.unique_id]
        condition = len(features) > 0 and example.par_idx == 0 and feature.span_idx == 0

        if condition:
            in_ = (features, results)
            inqueue.put(in_)
            prev_ex = id2example[results[0].unique_id]
            if prev_ex.doc_idx % 200 == 0:
                logger.info(f'saving {len(features)} features from doc {prev_ex.title} (doc_idx: {prev_ex.doc_idx})')
                logger.info(
                    '[%d/%d at %.1f second] ' % (count + 1, len(all_features), time() - start_time) +
                    '[inqueue, outqueue size: %d vs %d]' % (inqueue.qsize(), outqueue.qsize())
                )
            features = [feature]
            results = [result]
        else:
            features.append(feature)
            results.append(result)
    in_ = (features, results)
    inqueue.put(in_)
    for _ in range(NUM_THREAD):
        inqueue.put(None)

    for in_p in in_p_list:
        in_p.join()
    for out_p in out_p_list:
        out_p.join()

    b_stats = collections.OrderedDict(sorted(b_quant_stat.items()))
    stats = collections.OrderedDict(sorted(quant_stat.items()))
    for k, v in b_stats.items():
        print(k, v)
    for k, v in stats.items():
        print(k, v)


def write_filter(all_examples, all_features, all_results, tokenizer, hdf5_path, filter_threshold, verbose_logging, has_title=False):

    assert len(all_examples) > 0
    id2feature = {feature.unique_id: feature for feature in all_features}
    id2example_ = {id_: all_examples[id2feature[id_].example_index] for id_ in id2feature}

    def add(inqueue_, outqueue_):
        for item in iter(inqueue_.get, None):
            args = list(item[:2]) + [
                None, None, tokenizer, verbose_logging, has_title, filter_threshold
            ]
            out = pool_func(args)
            outqueue_.put(out)
        outqueue_.put(None)

    def write(outqueue_):
        with h5py.File(hdf5_path, 'a') as f:
            while True:
                metadata = outqueue_.get()
                if metadata:
                    did = str(metadata['did'])
                    if did in f:
                        logger.info('%s exists; replacing' % did)
                        del f[did]
                    dg = f.create_group(did)

                    dg.attrs['title'] = metadata['title']
                    dg.create_dataset('filter_start', data=metadata['filter_start'])
                    dg.create_dataset('filter_end', data=metadata['filter_end'])
                else:
                    break

    features = []
    results = []
    inqueue = Queue(maxsize=50)
    outqueue = Queue(maxsize=50)
    NUM_THREAD = 10
    in_p_list = [Process(target=add, args=(inqueue, outqueue)) for _ in range(NUM_THREAD)]
    out_p_list = [Thread(target=write, args=(outqueue,)) for _ in range(NUM_THREAD)]
    global id2example
    id2example = id2example_
    for in_p in in_p_list:
        in_p.start()
    for out_p in out_p_list:
        out_p.start()

    start_time = time()
    for count, result in enumerate(tqdm(all_results, total=len(all_features))):
        example = id2example[result.unique_id]
        feature = id2feature[result.unique_id]
        condition = len(features) > 0 and example.par_idx == 0 and feature.span_idx == 0

        if condition:
            # print('put')
            # in_ = (id2example_, features, results)
            in_ = (features, results)
            inqueue.put(in_)
            # import pdb; pdb.set_trace()
            prev_ex = id2example[results[0].unique_id]
            if prev_ex.doc_idx % 200 == 0:
                logger.info(f'saving {len(features)} features from doc {prev_ex.title} (doc_idx: {prev_ex.doc_idx})')
                logger.info(
                    '[%d/%d at %.1f second] ' % (count + 1, len(all_features), time() - start_time) +
                    '[inqueue, outqueue size: %d vs %d]' % (inqueue.qsize(), outqueue.qsize())
                )
            features = [feature]
            results = [result]
        else:
            features.append(feature)
            results.append(result)
    in_ = (features, results)
    inqueue.put(in_)
    for _ in range(NUM_THREAD):
        inqueue.put(None)

    for in_p in in_p_list:
        in_p.join()
    for out_p in out_p_list:
        out_p.join()


def get_question_results(question_examples, query_eval_features, question_dataloader, device, model, batch_size):
    id2feature = {feature.unique_id: feature for feature in query_eval_features}
    id2example = {id_: question_examples[id2feature[id_].example_index] for id_ in id2feature}

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    for batch in tqdm(question_dataloader, desc="Evaluating", disable=True):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        assert len(batch) == 4

        with torch.no_grad():
            inputs = {
                "input_ids_": batch[0],
                "attention_mask_": batch[1],
                "token_type_ids_": batch[2],
                "return_query": True,
            }
            feature_indices = batch[3]
            assert len(feature_indices.size()) > 0
            # feature_indices.unsqueeze_(0)

            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = query_eval_features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [
                to_numpy(output[i]) if type(output) != dict else {k: to_numpy(v[i]) for k, v in output.items()}
                for output in outputs
            ]

            if len(output) != 2:
                raise NotImplementedError
            else:
                start_vec, end_vec = output
                result = QuestionResult(
                    unique_id,
                    qas_id=id2example[unique_id].qas_id,
                    input_ids=id2feature[unique_id].input_ids_,
                    start_vec=start_vec,
                    end_vec=end_vec,
                )
            yield result


def get_cq_results(examples, eval_features, dataloader, device, model, batch_size):
    id2feature = {feature.unique_id: feature for feature in eval_features}
    id2example = {id_: examples[id2feature[id_].example_index] for id_ in id2feature}

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    def to_list(tensor):
        return tensor.detach().cpu().tolist()

    for batch in tqdm(dataloader, desc="Evaluating", disable=True):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "input_ids_": batch[6],
                "attention_mask_": batch[7],
                "token_type_ids_": batch[8],
                "title_offset": batch[9],
            }
            feature_indices = batch[3]
            assert len(feature_indices.size()) > 0

            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = eval_features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            if len(output) != 4:
                raise NotImplementedError
            else:
                start_logits, end_logits, sft_logits, eft_logits = output
                result = SquadResult(
                    unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits,
                    sft_logits=sft_logits,
                    eft_logits=eft_logits,
                )
            yield result


def get_bertqa_results(examples, eval_features, dataloader, device, model, batch_size):
    id2feature = {feature.unique_id: feature for feature in eval_features}
    id2example = {id_: examples[id2feature[id_].example_index] for id_ in id2feature}

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    def to_list(tensor):
        return tensor.detach().cpu().tolist()

    for batch in tqdm(dataloader, desc="Evaluating", disable=True):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            feature_indices = batch[3]
            assert len(feature_indices.size()) > 0

            outputs = model[0](**inputs)
            outputs = model[1](outputs[0]).split(dim=2, split_size=1)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = eval_features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i].squeeze(1)) for output in outputs]

            if len(output) != 2:
                raise NotImplementedError
            else:
                start_logits, end_logits = output
                result = SquadResult(
                    unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits,
                    sft_logits=start_logits,
                    eft_logits=end_logits,
                )
            yield result
