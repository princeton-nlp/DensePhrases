import random
import torch
import numpy as np


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
