# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import pprint
import re
import string
from rouge import Rouge

from collections import Counter
from collections import defaultdict, OrderedDict

from . import kilt_utils
import pdb

# utility to get gold answers
def get_gold_answers(gold):
    ground_truths = set()
    for item in gold["output"]:
        if "answer" in item and item["answer"] and len(item["answer"].strip()) > 0:
            ground_truths.add(item["answer"].strip())
    return ground_truths


# utility to get max
def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# answer nomalization
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# F1 score definition
def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# EM score definition
def _exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# ROUGEL score definition
def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


def _calculate_metrics(gold_records, guess_records):

    assert len(gold_records) == len(
        guess_records
    ), "different size gold: {} guess: {}".format(len(gold_records), len(guess_records))

    total_count = 0

    # downstream metrics
    accuracy = 0
    normalized_em = 0
    normalized_f1 = 0
    rougel = 0

    # kilt metrics
    kilt_accuracy = 0
    kilt_em = 0
    kilt_f1 = 0
    kilt_rougel = 0

    for guess_item, gold_item in zip(guess_records, gold_records):

        # check ids
        assert (
            str(gold_item["id"]).strip() == str(guess_item["id"]).strip()
        ), "Items must have same order with same IDs"

        total_count += 1
        # check if each output of guess file exist in set of candidate answers
        gold_candidate_answers = get_gold_answers(gold_item)

        conditions = (len(guess_item["output"]) == 1) and (
            "answer" in guess_item["output"][0]
        )
        assert (
            conditions
        ), f"you should provide exactly one valid answer for {guess_item['id']}"
        guess_answer = str(guess_item["output"][0]["answer"]).strip()

        if len(guess_answer) == 0:
            # empty answer
            continue

        # 0. accuracy = strict exact match
        local_accuracy = 0
        if guess_answer in gold_candidate_answers:
            local_accuracy = 1
        accuracy += local_accuracy

        # 1. normalized exact match
        local_em = _metric_max_over_ground_truths(
            _exact_match_score, guess_answer, gold_candidate_answers
        )
        normalized_em += local_em

        # 2. normalized f1
        local_f1 = _metric_max_over_ground_truths(
            _f1_score, guess_answer, gold_candidate_answers
        )
        normalized_f1 += local_f1

        # 3. rougel
        local_rougel = _metric_max_over_ground_truths(
            _rougel_score, guess_answer, gold_candidate_answers
        )
        rougel += local_rougel

        # KILT-metrics
        Rprec = rprecision(
            guess_item, gold_item, rank_keys=["wikipedia_id"]
        )
        if Rprec == 1:
            # 1. KILT-AC
            kilt_accuracy += local_accuracy

            # 2. KILT-EM
            kilt_em += local_em

            # 3. KILT-F1
            kilt_f1 += local_f1

            # 4. KILT-RL
            kilt_rougel += local_rougel

    if total_count > 0:
        accuracy /= total_count
        normalized_em /= total_count
        normalized_f1 /= total_count
        rougel /= total_count
        kilt_accuracy /= total_count
        kilt_em /= total_count
        kilt_f1 /= total_count
        kilt_rougel /= total_count

    return {
        "kilt": {
            "KILT-accuracy": kilt_accuracy,
            "KILT-em": kilt_em,
            "KILT-f1": kilt_f1,
            "KILT-rougel": kilt_rougel,
        },
        "downstream": {
            "accuracy": accuracy,
            "em": normalized_em,
            "f1": normalized_f1,
            "rougel": rougel,
        },
    }


def validate_input(gold_records, guess_records):
    if len(gold_records) != len(guess_records):
        print(
            "WARNING: DIFFERENT SIZE gold: {} guess: {}".format(
                len(gold_records), len(guess_records)
            )
        )

    # align order
    gold_ids = []
    for gold in gold_records:
        assert str(gold["id"]).strip() not in gold_ids, "Gold IDs should be unique"
        gold_ids.append(str(gold["id"]).strip())

    id2gold_record = {}
    for gold in gold_records:
        assert (
            str(gold["id"]).strip() not in id2gold_record
        ), "Golden IDs should be unique"
        id2gold_record[str(gold["id"]).strip()] = gold
        
    id2guess_record = {}
    for guess in guess_records:
        assert (
            str(guess["id"]).strip() not in id2guess_record
        ), "Prediction IDs should be unique"
        id2guess_record[str(guess["id"]).strip()] = guess
    
    guess_records = []
    gold_records = []
    
    num_mismatched = 0
    for id in gold_ids:
        if id in id2guess_record:
            guess_records.append(id2guess_record[id])
            gold_records.append(id2gold_record[id])
        else:
            num_mismatched +=1
            
    if num_mismatched>0:
        print(f"WARNING: {num_mismatched} predictions were not provided")
            
    return gold_records, guess_records


def evaluate(gold, guess):
    pp = pprint.PrettyPrinter(indent=4)

    gold_records = kilt_utils.load_data(gold)
    guess_records = kilt_utils.load_data(guess)

    # 0. validate input
    gold_records, guess_records = validate_input(gold_records, guess_records)

    # 1. downstream + kilt
    result = _calculate_metrics(gold_records, guess_records)

    # 2. retrieval performance
    retrieval_results = compute(
        gold_records, guess_records, ks=[1, 5], rank_keys=["wikipedia_id"]
    )
    result["retrieval"] = {
        "Rprec": retrieval_results["Rprec"],
        "recall@5": retrieval_results["recall@5"],
    }

    # pp.pprint(result)
    return result

def _remove_duplicates(obj):
    obj_tmp = []
    for o in obj:
        if o not in obj_tmp:
            obj_tmp.append(o)
    return obj_tmp


def _get_ids_list(datapoint, rank_keys, verbose=False):
    # collect all gold ids
    ids_list = []
    for output in datapoint["output"]:
        current_ids_list = []
        if "provenance" in output:
            for provenance in output["provenance"]:
                if any(rank_key not in provenance for rank_key in rank_keys):
                    missing = set(rank_keys) - set(
                        list(provenance.keys())
                    ).intersection(set(rank_keys))
                    if verbose:
                        print(
                            f"WARNING: missing key(s) {missing} in provenance, unable to compute retrieval for those."
                        )
                else:
                    current_ids_list.append(
                        "+".join(
                            [
                                str(provenance[rank_key]).strip()
                                for rank_key in rank_keys
                            ]
                        )
                    )
        ids_list.append(_remove_duplicates(current_ids_list))  # remove duplicates

    # consider only unique ids
    return ids_list


def get_rank(guess_item, gold_item, k, rank_keys, verbose=False):
    """
    The main idea is to consider each evidence set as a single point in the rank.
    The score in the rank for an evidence set is given by the lowest scored evidence in the set.
    """

    assert k > 0, "k must be a positive integer grater than 0."

    rank = []
    num_distinct_evidence_sets = 0

    guess_ids = _get_ids_list(guess_item, rank_keys)[0]

    if guess_ids and len(guess_ids) > 0:

        # 1. collect evidence sets and their sizes
        evidence_sets = []
        e_size = defaultdict(int)
        for output in gold_item["output"]:
            if "provenance" in output:
                e_set = {
                    "+".join(
                        [str(provenance[rank_key]).strip() for rank_key in rank_keys]
                    )
                    for provenance in output["provenance"]
                }
                if e_set not in evidence_sets:  # no duplicate evidence set
                    evidence_sets.append(e_set)
                    e_size[len(e_set)] += 1
        num_distinct_evidence_sets = len(evidence_sets)

        # 2. check what's the minimum number of predicted pages needed to get a robust P/R@k
        min_prediction_size = 0
        c = 0
        for size, freq in sorted(e_size.items(), reverse=True):
            for _ in range(freq):
                min_prediction_size += size
                c += 1
                if c == k:
                    break
            if c == k:
                break
        # if the number of evidence sets is smaller than k
        min_prediction_size += k - c

        if verbose and len(guess_ids) < min_prediction_size:
            print(
                f"WARNING: you should provide at least {min_prediction_size} provenance items for a robust recall@{k} computation (you provided {len(guess_ids)} item(s))."
            )

        # 3. rank by gruping pages in each evidence set (each evidence set count as 1),
        # the position in the rank of each evidence set is given by the last page in guess_ids
        # non evidence pages counts as 1
        rank = []
        for guess_id in guess_ids:
            guess_id = str(guess_id).strip()
            found = False
            for idx, e_set in enumerate(evidence_sets):

                e_set_id = f"evidence_set:{idx}"

                if guess_id in e_set:
                    found = True

                    # remove from the rank previous points referring to this evidence set
                    if e_set_id in rank:
                        rank.remove(e_set_id)

                    # remove the guess_id from the evidence set
                    e_set.remove(guess_id)

                    if len(e_set) == 0:
                        # it was the last evidence, it counts as true in the rank
                        rank.append(True)
                    else:
                        # add a point for this partial evidence set
                        rank.append(e_set_id)

            if not found:
                rank.append(False)

    return rank, num_distinct_evidence_sets


# 1. Precision computation
def _precision_at_k(rank, k):

    # precision @ k
    p = rank[:k].count(True) / k

    return p


# 2. Recall computation
def _recall_at_k(rank, num_distinct_evidence_sets, k):

    r = rank[:k].count(True) / num_distinct_evidence_sets

    return r


# 3. Success rate computation
def _success_rate_at_k(rank, k):

    # success rate @ k
    p = int(True in rank[:k])

    return p


def _computeRprec(guess_ids, gold_ids):

    R = len(gold_ids)
    num = 0

    for prediction in guess_ids[:R]:
        if str(prediction).strip() in gold_ids:
            num += 1

    Rprec = num / R if R > 0 else 0
    return Rprec


# R-precision https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-39940-9_486
def rprecision(guess_item, gold_item, rank_keys):
    gold_ids_list = _get_ids_list(gold_item, rank_keys)
    guess_ids = _get_ids_list(guess_item, rank_keys)[0]
    Rprec_vector = []
    for gold_ids in gold_ids_list:
        Rprec = _computeRprec(guess_ids, gold_ids)
        Rprec_vector.append(Rprec)
    return max(Rprec_vector)


def get_ranking_metrics(guess_item, gold_item, ks, rank_keys):

    Rprec = 0
    P_at_k = {"precision@{}".format(k): 0 for k in sorted(ks) if k > 0}
    R_at_k = {"recall@{}".format(k): 0 for k in sorted(ks) if k > 1}
    S_at_k = {"success_rate@{}".format(k): 0 for k in sorted(ks) if k > 1}

    assert (
        "output" in guess_item and len(guess_item["output"]) == 1
    ), f"guess should provide exactly one output for {guess_item['id']}"

    Rprec = rprecision(guess_item, gold_item, rank_keys=rank_keys)
    for k in ks:

        # 0. get rank
        rank, num_distinct_evidence_sets = get_rank(
            guess_item, gold_item, k, rank_keys=rank_keys
        )

        if num_distinct_evidence_sets > 0:

            # 1. precision
            P_at_k["precision@{}".format(k)] = _precision_at_k(rank, k)

            # 2. recall
            R_at_k["recall@{}".format(k)] = _recall_at_k(
                rank, num_distinct_evidence_sets, k
            )

            # 3. success rate
            S_at_k["success_rate@{}".format(k)] = _success_rate_at_k(rank, k)

        # else:
        #     print(
        #         "WARNING: the number of distinct evidence sets is 0 for {}".format(
        #             gold_item
        #         )
        #     )

    return {"Rprec": Rprec, **P_at_k, **R_at_k, **S_at_k}


def compute(gold_dataset, guess_dataset, ks, rank_keys):

    ks = sorted([int(x) for x in ks])

    result = OrderedDict()
    result["Rprec"] = 0.0
    for k in ks:
        if k > 0:
            result["precision@{}".format(k)] = 0.0
        if k > 1:
            result["recall@{}".format(k)] = 0.0
            result["success_rate@{}".format(k)] = 0.0

    assert len(guess_dataset) == len(
        gold_dataset
    ), "different size gold: {} guess: {}".format(len(guess_dataset), len(gold_dataset))

    for gold, guess in zip(guess_dataset, gold_dataset):
        assert (
            str(gold["id"]).strip() == str(guess["id"]).strip()
        ), "Items must have same order with same IDs"

    for guess_item, gold_item in zip(guess_dataset, gold_dataset):
        ranking_metrics = get_ranking_metrics(guess_item, gold_item, ks, rank_keys)
        result["Rprec"] += ranking_metrics["Rprec"]
        for k in ks:
            if k > 0:
                result["precision@{}".format(k)] += ranking_metrics[
                    "precision@{}".format(k)
                ]
            if k > 1:
                result["recall@{}".format(k)] += ranking_metrics["recall@{}".format(k)]
                result["success_rate@{}".format(k)] += ranking_metrics[
                    "success_rate@{}".format(k)
                ]

    if len(guess_dataset) > 0:
        result["Rprec"] /= len(guess_dataset)
        for k in ks:
            if k > 0:
                result["precision@{}".format(k)] /= len(guess_dataset)
            if k > 1:
                result["recall@{}".format(k)] /= len(guess_dataset)
                result["success_rate@{}".format(k)] /= len(guess_dataset)

    return result
