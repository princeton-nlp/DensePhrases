""" Very heavily inspired by the official evaluation script for SQuAD version 2.0 which was
modified by XLNet authors to update `find_best_threshold` scripts for SQuAD V2.0

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""


import collections
import logging
import tqdm
import h5py
import numpy as np
import torch

from functools import partial
from multiprocessing import Queue, Process, Pool, cpu_count
from threading import Thread
from time import time
from tqdm import tqdm
from .single_utils import to_list, ForkedPdb
from .file_utils import is_torch_available
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset


logger = logging.getLogger(__name__)
# For debugging
quant_stat = {}
b_quant_stat = {}

fid2example = None
eid2fids = None


def get_metadata(features, results, max_answer_length, tokenizer, has_title):
    global fid2example, eid2fids

    assert len(features) == len(results)

    # Get rid of titles + save start only (as start and end are shared)
    roberta_add = 1 if "roberta" in str(type(tokenizer)) else 0
    toffs = [(f['input_ids'].index(tokenizer.sep_token_id))*int(has_title) + roberta_add for f in features]

    # Filter reps
    fs = np.concatenate(
        [result['filter_start'][to+1:sum(feature['attention_mask']) - 1] for feature, result, to in zip(features, results, toffs)], axis=0
    )
    fe = np.concatenate(
        [result['filter_end'][to+1:sum(feature['attention_mask']) - 1] for feature, result, to in zip(features, results, toffs)], axis=0
    )

    if max_answer_length is None:
        example = fid2example[results[-1]['feature_id']]
        metadata = {
            'did': example['doc_idx'], 'title': example['title'],
            'filter_start': fs, 'filter_end': fe
        }
        return metadata

    # start vectors
    start = np.concatenate(
        [result['start'][to+1:sum(feature['attention_mask']) - 1] for feature, result, to in zip(features, results, toffs)],
        axis=0
    )
    len_per_para = [len(f['input_ids'][to+1:sum(f['attention_mask'])-1]) for to, f in zip(toffs, features)]

    # Start2end map
    start2end = -1 * np.ones([np.shape(start)[0], max_answer_length], dtype=np.int32)
    idx = 0
    for feature, result, to in zip(features, results, toffs):
        for i in range(to+1, sum(feature['attention_mask']) - 1):
            for j in range(i, min(i + max_answer_length, sum(feature['attention_mask']) - 1)):
                start2end[idx, j - i] = idx + j - i
            idx += 1

    word2char_start = np.zeros([start.shape[0]], dtype=np.int32)
    word2char_end = np.zeros([start.shape[0]], dtype=np.int32)

    # Orig map
    sep = ' [PAR] '
    full_text = ""
    prev_example = None
    # ForkedPdb().set_trace()
    word_pos = 0
    for f_idx, (feature, to) in enumerate(zip(features, toffs)):
        result = results[f_idx]
        example = fid2example[result['feature_id']]
        if prev_example is not None and eid2fids[result['example_id']][0] == result['feature_id']:
            full_text = full_text + prev_example['context'] + sep

        for i in range(to+1, sum(feature['attention_mask']) - 1):
            word2char_start[word_pos] = feature['offset_mapping'][i][0] + len(full_text)
            word2char_end[word_pos] = feature['offset_mapping'][i][1] + len(full_text)
            word_pos += 1
        prev_example = example
    full_text = full_text + prev_example['context']

    metadata = {
        'did': prev_example['doc_idx'], 'context': full_text, 'title': prev_example['title'],
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

    if len(all_idxs) == 0:
        all_idxs = np.where(metadata['filter_start'] > -999999)[0][:1] # just get all
        end_long2short = {long: short for short, long in enumerate(all_idxs)}
        print('all idxs were filtered, so use only one vector for this:', len(all_idxs))
    metadata['start'] = metadata['start'][all_idxs] # union of start/end
    metadata['f2o_start'] = all_idxs
    metadata['start2end'] = metadata['start2end'][all_idxs]
    
    for i, each in enumerate(metadata['start2end']):
        for j, long in enumerate(each.tolist()):
            metadata['start2end'][i, j] = end_long2short[long] if long in end_long2short else -1

    return metadata


def float_to_int8(num, offset, factor):
    out = (num - offset) * factor
    out = out.clip(-128, 127)
    out = np.round(out).astype(np.int8)
    return out


def int8_to_float(num, offset, factor):
    return num.astype(np.float32) / factor + offset


def compress_metadata(metadata, dense_offset, dense_scale, record_histogram=False):
    for key in ['start']:
        if key in metadata:
            
            if record_histogram and (key == 'start'):
                for meta in metadata[key]:
                    for number in meta:
                        num_str = "%.1f" % number
                        if float(num_str) not in b_quant_stat:
                            b_quant_stat[float(num_str)] = 0
                        b_quant_stat[float(num_str)] += 1
            
            metadata[key] = float_to_int8(metadata[key], dense_offset, dense_scale)
            
            if record_histogram and (key == 'start'):
                for meta in metadata[key]:
                    for number in meta:
                        num_str = "%d" % number
                        if int(num_str) not in quant_stat:
                            quant_stat[int(num_str)] = 0
                        quant_stat[int(num_str)] += 1
            
    return metadata


def pool_func(item):
    metadata_ = get_metadata(*item[:-1])
    if 'start' in metadata_:
        metadata_ = filter_metadata(metadata_, item[-1])
    return metadata_


def write_phrases(all_examples, all_features, all_results, tokenizer, output_dump_file, offset, args):
    global fid2example, eid2fids
    assert len(all_examples) > 0

    # unique_id to feature, example
    fid2feature = {id: feature for id, feature in enumerate(all_features)}
    fid2example = {id: all_examples[fid2feature[id]['example_id']] for id in fid2feature}
    eid2fids = collections.defaultdict(list)
    for i, feature in enumerate(all_features):
        eid2fids[feature["example_id"]].append(i)

    def add(inqueue_, outqueue_):
        for item in iter(inqueue_.get, None):
            # start_time = time()
            new_item = list(item[:2]) + [
                args.max_answer_length, tokenizer,
                args.append_title, args.filter_threshold
            ]
            out = pool_func(new_item)
            # print(f'in {time() - start_time:.1f} sec, {inqueue_.qsize()}')
            outqueue_.put(out)

        outqueue_.put(None)

    def write(outqueue_):
        with h5py.File(output_dump_file, 'a') as f:
            while True:
                metadata = outqueue_.get()
                if metadata:
                    # start_time = time()
                    did = str(offset + metadata['did'])
                    if did in f:
                        logger.info('%s exists; replacing' % did)
                        del f[did]
                        # logger.info('%s exists; skipping' % did)
                        # continue
                    dg = f.create_group(did)

                    dg.attrs['context'] = metadata['context']
                    dg.attrs['title'] = metadata['title']
                    if args.dense_offset is not None:
                        metadata = compress_metadata(metadata, args.dense_offset, args.dense_scale)
                        dg.attrs['offset'] = args.dense_offset
                        dg.attrs['scale'] = args.dense_scale
                    dg.create_dataset('start', data=metadata['start'])
                    dg.create_dataset('len_per_para', data=metadata['len_per_para'])
                    dg.create_dataset('start2end', data=metadata['start2end'])
                    dg.create_dataset('word2char_start', data=metadata['word2char_start'])
                    dg.create_dataset('word2char_end', data=metadata['word2char_end'])
                    dg.create_dataset('f2o_start', data=metadata['f2o_start'])
                    # print(f'out {time() - start_time:.1f} sec, {outqueue_.qsize()} ')

                    # write filters if necessary
                    # dg.create_dataset('filter_start', data=metadata['filter_start'])
                    # dg.create_dataset('filter_end', data=metadata['filter_end'])
                else:
                    break

    features = []
    results = []
    inqueue = Queue(maxsize=50)
    outqueue = Queue(maxsize=50)
    NUM_THREAD = 50
    in_p_list = [Process(target=add, args=(inqueue, outqueue)) for _ in range(NUM_THREAD)]
    out_p_list = [Thread(target=write, args=(outqueue,)) for _ in range(NUM_THREAD)]
    
    for in_p in in_p_list:
        in_p.start()
    for out_p in out_p_list:
        out_p.start()

    start_time = time()
    for count, result in enumerate(tqdm(all_results, total=len(all_features))):
        example = fid2example[result['feature_id']]
        feature = fid2feature[result['feature_id']]
        condition = (
            len(features) > 0 and example['par_idx'] == 0 \
            and eid2fids[result['example_id']][0] == result['feature_id'] # First span
        )
        assert result['feature_id'] == count, (result['feature_id'], count)

        if condition:
            in_ = (features, results)
            inqueue.put(in_)
            prev_ex = fid2example[results[0]['feature_id']]
            if prev_ex['doc_idx'] % 200 == 0:
                logger.info(f"saving {len(features)} features from doc {prev_ex['title']} (doc_idx: {offset + prev_ex['doc_idx']})")
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


def convert_question_to_feature(example, max_query_length):
    # Query features
    feature = tokenizer(
        example['question_text'],
        max_length=max_query_length,
        return_overflowing_tokens=False,
        padding="max_length",
        truncation="only_first",
        return_token_type_ids=True, # TODO: check token_type_ids of questions
    )
    feature['qas_id'] = example['qas_id']
    feature['question_text'] = example['question_text']
    # logger.info(f'prepro 0) {time()-start_time}')
    return feature


def convert_question_to_feature_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_questions_to_features(
    examples,
    tokenizer,
    max_query_length,
    threads=1,
    tqdm_enabled=True,
):
    """
    convert questions to features
    """

    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    # start_time = time()
    if threads > 1:
        with Pool(threads, initializer=convert_question_to_feature_init, initargs=(tokenizer,)) as p:
            annotate_ = partial(
                convert_question_to_feature,
                max_query_length=max_query_length,
            )
            features = list(
                tqdm(
                    p.imap(annotate_, examples, chunksize=32),
                    total=len(examples),
                    desc="convert squad examples to features",
                    disable=not tqdm_enabled,
                )
            )
    else:
        convert_question_to_feature_init(tokenizer)
        features = [convert_question_to_feature(
            example,
            max_query_length=max_query_length,
        ) for example in examples]

    # logger.info(f'prepro 1) {time()-start_time}')
    new_features = []
    unique_id = 1000000000
    for feature in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        feature['unique_id'] = unique_id
        new_features.append(feature)
        unique_id += 1
    features = new_features
    del new_features
    
    if not is_torch_available():
        raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

    # Question-side features
    all_input_ids_ = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_attention_masks_ = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
    all_token_type_ids_ = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)
    all_feature_index_ = torch.arange(all_input_ids_.size(0), dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids_, all_attention_masks_, all_token_type_ids_, all_feature_index_
    )
    return features, dataset


def get_question_dataloader(questions, tokenizer, max_query_length=64, batch_size=64):
    examples = [{'qas_id': q_idx, 'question_text': q} for q_idx, q in enumerate(questions)]
    features, dataset = convert_questions_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_query_length=max_query_length,
        threads=1,
        tqdm_enabled=False,
    )
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

    return eval_dataloader, features


def get_question_results(query_eval_features, question_dataloader, device, model):

    for batch in tqdm(question_dataloader, desc="Evaluating", disable=True):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        assert len(batch) == 4

        with torch.inference_mode():
            inputs = {
                "input_ids_": batch[0],
                "attention_mask_": batch[1],
                "token_type_ids_": batch[2],
            }
            feature_indices = batch[3]
            assert len(feature_indices.size()) > 0
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = query_eval_features[feature_index.item()]
            output = [
                to_list(output[i]) if type(output) != dict else {k: to_list(v[i]) for k, v in output.items()}
                for output in outputs
            ]
            start_vec, end_vec = output
            result = (start_vec, end_vec, eval_feature['question_text'])
            yield result