# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import json
import argparse
import logging
import os
import copy
import torch
import numpy as np
import collections
import hydra
from torch import nn
from typing import List, Tuple
from omegaconf import DictConfig, OmegaConf
from dpr.data.biencoder_data import BiEncoderPassage
from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger

from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)

from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from densephrases.utils.squad_utils import get_cq_dataloader, SquadResult
from densephrases.utils.single_utils import set_seed, to_list, to_numpy, backward_compat


logger = logging.getLogger(__name__)
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)
print(set([k.split('-')[0] for k in ALL_MODELS]))


def gen_ctx_vectors(
    cfg: DictConfig,
    ctx_rows: List[Tuple[object, BiEncoderPassage]],
    q_rows: List[object],
    model: nn.Module,
    tensorizer: Tensorizer,
    insert_title: bool = True,
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = cfg.batch_size
    total = 0
    results = []
    for j, batch_start in enumerate(range(0, n, bsz)):
        # Passage preprocess # TODO; max seq length check
        batch = ctx_rows[batch_start : batch_start + bsz]
        batch_token_tensors = [
            tensorizer.text_to_tensor(
                ctx[1].text, title=ctx[1].title if insert_title else None
            )
            for ctx in batch
        ]

        ctx_ids_batch = move_to_device(
            torch.stack(batch_token_tensors, dim=0), cfg.device
        )
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), cfg.device)
        ctx_attn_mask = move_to_device(
            tensorizer.get_attn_mask(ctx_ids_batch), cfg.device
        )

        # Question preprocess
        q_batch = q_rows[batch_start : batch_start + bsz]
        q_batch_token_tensors = [
            tensorizer.text_to_tensor(
                qq
            )
            for qq in q_batch
        ]

        q_ids_batch = move_to_device(
            torch.stack(q_batch_token_tensors, dim=0), cfg.device
        )
        q_seg_batch = move_to_device(torch.zeros_like(q_ids_batch), cfg.device)
        q_attn_mask = move_to_device(
            tensorizer.get_attn_mask(q_ids_batch), cfg.device
        )

        # Selector
        from dpr.data.biencoder_data import DEFAULT_SELECTOR
        selector = DEFAULT_SELECTOR
        rep_positions = selector.get_positions(
            q_ids_batch, tensorizer
        )

        with torch.no_grad():
            q_dense, ctx_dense = model(
                q_ids_batch, q_seg_batch, q_attn_mask, ctx_ids_batch, ctx_seg_batch, ctx_attn_mask,
                representation_token_pos=rep_positions,
            )
        q_dense = q_dense.cpu()
        ctx_dense = ctx_dense.cpu()
        ctx_ids = [r[0] for r in batch]

        assert len(ctx_ids) == q_dense.size(0) == ctx_dense.size(0)
        total += len(ctx_ids)

        results.extend(
            [(ctx_ids[i], q_dense[i].numpy(), ctx_dense[i].numpy(), q_dense[i].numpy().dot(ctx_dense[i].numpy()))
             for i in range(q_dense.size(0))]
        )

        if total % 10 == 0:
            logger.info("Encoded questions / passages %d", total)
            # break
    return results


def dump_passages(cfg, encoder, tensorizer):

    logger.info(f"Pair file from {cfg.pair_file}")
    passage_pairs = json.load(open(cfg.pair_file))['data']
    
    questions = [pp['question'] for pp in passage_pairs]
    answers = [pp['answer'][0] for pp in passage_pairs]
    answers_start = [pp['answer'][1] for pp in passage_pairs]
    titles = [pp['title'] for pp in passage_pairs]
    gold_passages = [pp['gold_passage'] for pp in passage_pairs]
    entail_neg_passages = [pp['entail_neg_passage'] for pp in passage_pairs]
    neg_titles = [pp['neg_title'] for pp in passage_pairs]
    topic_neg_passages = [pp['topic_neg_passage'] for pp in passage_pairs]
    
    stats = {}
    logger.info(f"***** Processing gold passages *****")
    gold_passages = [
        (p_id, BiEncoderPassage(title=title, text=passage)) for p_id, (title, passage) in enumerate(zip(titles, gold_passages))
    ]
    data = gen_ctx_vectors(cfg, gold_passages, questions, encoder, tensorizer, True)
    for p_id, _, _, gold_score in data:
        if p_id not in stats:
            stats[p_id] = {'gold_score': -1e9, 'topic_neg_score': -1e9, 'entail_neg_score': -1e9}
        stats[p_id]['gold_score'] = gold_score if gold_score > stats[p_id]['gold_score'] else stats[p_id]['gold_score']
        # break

    logger.info(f"***** Processing topic neg passages *****")
    topic_neg_passages = [
        (p_id, BiEncoderPassage(title=title, text=passage)) for p_id, (title, passage) in enumerate(zip(neg_titles, topic_neg_passages))
    ]
    data = gen_ctx_vectors(cfg, topic_neg_passages, questions, encoder, tensorizer, True)
    for p_id, _, _, topic_neg_score in data:
        if p_id not in stats:
            continue
        stats[p_id]['topic_neg_score'] = topic_neg_score if topic_neg_score > stats[p_id]['topic_neg_score'] else stats[p_id]['topic_neg_score']
        # break

    logger.info(f"***** Processing entail neg passages *****")
    entail_neg_passages = [
        (p_id, BiEncoderPassage(title=title, text=passage)) for p_id, (title, passage) in enumerate(zip(titles, entail_neg_passages))
    ]
    data = gen_ctx_vectors(cfg, entail_neg_passages, questions, encoder, tensorizer, True)
    for p_id, _, _, entail_neg_score in data:
        if p_id not in stats:
            continue
        stats[p_id]['entail_neg_score'] = entail_neg_score if entail_neg_score > stats[p_id]['entail_neg_score'] else stats[p_id]['entail_neg_score']
        # break
    
    if not all(all(val > -999 for val in score.values()) for score in stats.values()):
        import pdb; pdb.set_trace()
    gold_mean = sum([stat['gold_score'] for stat in stats.values()]) / len(stats)
    topic_mean = sum([stat['topic_neg_score'] for stat in stats.values()]) / len(stats)
    entail_mean = sum([stat['entail_neg_score'] for stat in stats.values()]) / len(stats)

    L_topic = sum([
        -torch.nn.functional.log_softmax(torch.Tensor([stat['gold_score'], stat['topic_neg_score']]), dim=-1)[0]
        for stat in stats.values()
    ])/len(stats)
    L_hard = sum([
        -torch.nn.functional.log_softmax(torch.Tensor([stat['gold_score'], stat['entail_neg_score']]), dim=-1)[0]
        for stat in stats.values()
    ])/len(stats)

    logger.info(f'gold mean: {gold_mean:.2f}, topic mean: {topic_mean:.2f}, entail mean: {entail_mean:.2f}')
    logger.info(f'topical relevance: {gold_mean - topic_mean:.2f}, fine-grained entailment: {gold_mean - entail_mean:.2f}')
    logger.info(f'L_topic: {L_topic:.4f}, L_hard: {L_hard:.4f}')
    logger.info(f"Analysis done for {len(passage_pairs)} (processed={len(stats)})")


@hydra.main(config_path="dpr-conf", config_name="gen_embs")
def main(cfg: DictConfig):

    assert cfg.pair_file, "Please specify passages source as pair_file param"
    assert cfg.model_file, "Please specify encoder checkpoint as model_file param"

    cfg = setup_cfg_gpu(cfg)

    saved_state = load_states_from_checkpoint(cfg.model_file)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    logger.info("CFG:")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    tensorizer, encoder, _ = init_biencoder_components(
        cfg.encoder.encoder_model_type, cfg, inference_only=True, cache_dir='/n/fs/nlp-jl5167/cache'
    )

    # load weights from the model file
    logger.info("Loading saved model state ...")
    logger.debug("saved model keys =%s", saved_state.model_dict.keys())

    encoder.load_state_dict(saved_state.model_dict)
    encoder.to(cfg.device)

    # Set seed
    # set_seed(args)
    cfg.encoder.sequence_length = 512 # TODO: Passage can be cut by max_seq_length
    logger.info(f"Max seq length: {cfg.encoder.sequence_length}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Dump passages
    dump_passages(cfg, encoder, tensorizer)


if __name__ == "__main__":
    main()
