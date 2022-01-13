import logging
import os
import math
import pickle
import string
import torch

from functools import partial
from multiprocessing import Pool, cpu_count
from time import time
from tqdm import tqdm
from .file_utils import is_torch_available
from .data_utils import whitespace_tokenize
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset


logger = logging.getLogger(__name__)


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


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


class TrueCaser(object):
    def __init__(self, dist_file_path=None):
        """ Initialize module with default data/english.dist file """
        if dist_file_path is None:
            dist_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data/english_with_questions.dist")

        with open(dist_file_path, "rb") as distributions_file:
            pickle_dict = pickle.load(distributions_file)
            self.uni_dist = pickle_dict["uni_dist"]
            self.backward_bi_dist = pickle_dict["backward_bi_dist"]
            self.forward_bi_dist = pickle_dict["forward_bi_dist"]
            self.trigram_dist = pickle_dict["trigram_dist"]
            self.word_casing_lookup = pickle_dict["word_casing_lookup"]

    def get_score(self, prev_token, possible_token, next_token):
        pseudo_count = 5.0

        # Get Unigram Score
        nominator = self.uni_dist[possible_token] + pseudo_count
        denominator = 0
        for alternativeToken in self.word_casing_lookup[
                possible_token.lower()]:
            denominator += self.uni_dist[alternativeToken] + pseudo_count

        unigram_score = nominator / denominator

        # Get Backward Score
        bigram_backward_score = 1
        if prev_token is not None:
            nominator = (
                self.backward_bi_dist[prev_token + "_" + possible_token] +
                pseudo_count)
            denominator = 0
            for alternativeToken in self.word_casing_lookup[
                    possible_token.lower()]:
                denominator += (self.backward_bi_dist[prev_token + "_" +
                                                      alternativeToken] +
                                pseudo_count)

            bigram_backward_score = nominator / denominator

        # Get Forward Score
        bigram_forward_score = 1
        if next_token is not None:
            next_token = next_token.lower()  # Ensure it is lower case
            nominator = (
                self.forward_bi_dist[possible_token + "_" + next_token] +
                pseudo_count)
            denominator = 0
            for alternativeToken in self.word_casing_lookup[
                    possible_token.lower()]:
                denominator += (
                    self.forward_bi_dist[alternativeToken + "_" + next_token] +
                    pseudo_count)

            bigram_forward_score = nominator / denominator

        # Get Trigram Score
        trigram_score = 1
        if prev_token is not None and next_token is not None:
            next_token = next_token.lower()  # Ensure it is lower case
            nominator = (self.trigram_dist[prev_token + "_" + possible_token +
                                           "_" + next_token] + pseudo_count)
            denominator = 0
            for alternativeToken in self.word_casing_lookup[
                    possible_token.lower()]:
                denominator += (
                    self.trigram_dist[prev_token + "_" + alternativeToken +
                                      "_" + next_token] + pseudo_count)

            trigram_score = nominator / denominator

        result = (math.log(unigram_score) + math.log(bigram_backward_score) +
                  math.log(bigram_forward_score) + math.log(trigram_score))

        return result

    def first_token_case(self, raw):
        return f'{raw[0].upper()}{raw[1:]}'

    def get_true_case(self, sentence, out_of_vocabulary_token_option="title"):
        """ Returns the true case for the passed tokens.
        @param tokens: Tokens in a single sentence
        @param outOfVocabulariyTokenOption:
            title: Returns out of vocabulary (OOV) tokens in 'title' format
            lower: Returns OOV tokens in lower case
            as-is: Returns OOV tokens as is
        """
        tokens = whitespace_tokenize(sentence)

        tokens_true_case = []
        for token_idx, token in enumerate(tokens):

            if token in string.punctuation or token.isdigit():
                tokens_true_case.append(token)
            else:
                token = token.lower()
                if token in self.word_casing_lookup:
                    if len(self.word_casing_lookup[token]) == 1:
                        tokens_true_case.append(
                            list(self.word_casing_lookup[token])[0])
                    else:
                        prev_token = (tokens_true_case[token_idx - 1]
                                      if token_idx > 0 else None)
                        next_token = (tokens[token_idx + 1]
                                      if token_idx < len(tokens) - 1 else None)

                        best_token = None
                        highest_score = float("-inf")

                        for possible_token in self.word_casing_lookup[token]:
                            score = self.get_score(prev_token, possible_token,
                                                   next_token)

                            if score > highest_score:
                                best_token = possible_token
                                highest_score = score

                        tokens_true_case.append(best_token)

                    if token_idx == 0:
                        tokens_true_case[0] = self.first_token_case(tokens_true_case[0])

                else:  # Token out of vocabulary
                    if out_of_vocabulary_token_option == "title":
                        tokens_true_case.append(token.title())
                    elif out_of_vocabulary_token_option == "lower":
                        tokens_true_case.append(token.lower())
                    else:
                        tokens_true_case.append(token)

        return "".join([
            " " +
            i if not i.startswith("'") and i not in string.punctuation else i
            for i in tokens_true_case
        ]).strip()