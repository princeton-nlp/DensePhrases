import random
import torch
import logging
import copy
import os
import numpy as np

from functools import partial
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
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


def load_encoder(device, args, phrase_only=False):
    # Configure paths for DnesePhrases
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.pretrained_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.pretrained_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
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
        # TODO: need to update transformers so that from_pretrained maps to model hub directly
        if args.load_dir.startswith('princeton-nlp'):
            hf_model_path = f"https://huggingface.co/{args.load_dir}/resolve/main/pytorch_model.bin"
        else:
            hf_model_path = args.load_dir
        load_class = partial(
            Encoder.from_pretrained,
            pretrained_model_name_or_path=hf_model_path,
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
    )

    # Phrase only (for phrase embedding)
    if phrase_only:
        if hasattr(model, "module"):
            del model.module.query_start_encoder
            del model.module.query_end_encoder
        else:
            del model.query_start_encoder
            del model.query_end_encoder
        logger.info("Load only phrase encoders for embedding phrases")

    model.to(device)
    logger.info('Number of model parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))
    return model, tokenizer, config
