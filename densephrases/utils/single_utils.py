import random
import torch
import logging
import copy
import pdb
import sys
import numpy as np

from functools import partial
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForQuestionAnswering,
)
from densephrases import Encoder

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def backward_compat(model_dict):
    # Remove teacher
    model_dict = {key: val for key, val in model_dict.items() if not key.startswith('cross_encoder')}
    model_dict = {key: val for key, val in model_dict.items() if not key.startswith('bert_qd')}
    model_dict = {key: val for key, val in model_dict.items() if not key.startswith('qa_outputs')}

    # Replace old names to current ones
    mapping = {
        'bert_start': 'phrase_encoder',
        'bert_q_start': 'query_start_encoder',
        'bert_q_end': 'query_end_encoder',
    }
    new_model_dict = {}
    for key, val in model_dict.items():
        for old_key, new_key in mapping.items():
            if key.startswith(old_key):
                new_model_dict[key.replace(old_key, new_key)] = val
            elif all(not key.startswith(old_k) for old_k in mapping.keys()):
                new_model_dict[key] = val

    return new_model_dict


def load_encoder(device, args, phrase_only=False, query_only=False, freeze_embedding=True):
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.pretrained_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.pretrained_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=True,
    )

    # Prepare PLM if not load_dir
    pretrained = None
    if not args.load_dir:
        pretrained = AutoModel.from_pretrained(
            args.pretrained_name_or_path,
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        load_class = Encoder
        logger.info(f'DensePhrases encoder initialized with {args.pretrained_name_or_path} ({pretrained.__class__})')
    else:
        load_class = partial(
            Encoder.from_pretrained,
            pretrained_model_name_or_path=args.load_dir,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        logger.info(f'DensePhrases encoder loaded from {args.load_dir}')

    # DensePhrases encoder object
    model = load_class(
        config=config,
        tokenizer=tokenizer,
        transformer_cls=MODEL_MAPPING[config.__class__],
        pretrained=copy.deepcopy(pretrained) if pretrained is not None else None,
        lambda_kl=getattr(args, 'lambda_kl', 0.0),
        lambda_neg=getattr(args, 'lambda_neg', 0.0),
        lambda_flt=getattr(args, 'lambda_flt', 0.0),
        pbn_size=getattr(args, 'pbn_size', 0.0),
        return_phrase=phrase_only,
        return_query=query_only,
    )
    
    # Load teacher for training (freeze)
    if getattr(args, 'lambda_kl', 0.0) > 0.0 and args.teacher_dir:
        model.cross_encoder = AutoModelForQuestionAnswering.from_pretrained(
            args.teacher_dir,
            from_tf=bool(".ckpt" in args.teacher_dir),
            config=config,
            cache_dir=args.cache_dir,
        )
        for param in model.cross_encoder.parameters():
            param.requires_grad = False

    # Phrase only (for phrase embedding)
    if phrase_only:
        if hasattr(model, "module"):
            del model.module.query_start_encoder
            del model.module.query_end_encoder
        else:
            del model.query_start_encoder
            del model.query_end_encoder
        logger.info("Load only phrase encoders for embedding phrases")
    
    # Query only (for query embedding)
    if query_only:
        if hasattr(model, "module"):
            del model.module.phrase_encoder
        else:
            del model.phrase_encoder
        logger.info("Load only query encoders for embedding queries")
    
    if freeze_embedding:
        for name, param in model.named_parameters():
            if name.endswith(".embeddings.word_embeddings.weight"):
                param.requires_grad = False
                logger.info(f'freezing {name}')

    model.to(device)
    logger.info('Number of model parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))
    return model, tokenizer, config


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin