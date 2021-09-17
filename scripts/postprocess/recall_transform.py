import json
import os
import sys
import argparse
import numpy as np
from spacy.lang.en import English
from tqdm import tqdm

nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))


def main(args):
    pred_file = os.path.join(args.model_dir, 'pred', args.pred_file)
    my_pred = json.load(open(pred_file))

    my_target = []
    avg_len = []
    for qid, pred in tqdm(enumerate(my_pred.values())):
        my_dict = {"id": str(qid), "question": None, "answers": [], "ctxs": []}

        # truncate
        pred = {key: val[:args.psg_top_k] if key in ['evidence', 'title', 'se_pos', 'prediction'] else val for key, val in pred.items()}

        # TODO: need to add id for predictions.pred in the future
        my_dict["question"] = pred["question"]
        my_dict["answers"] = pred["answer"]
        pred["title"] = [titles[0] for titles in pred["title"]]

        assert len(set(pred["evidence"])) == len(pred["evidence"]) == len(pred["title"]), "Should use opt2 for aggregation"
        # assert all(pr in evd for pr, evd in zip(pred["prediction"], pred["evidence"]))  # prediction included

        # Pad up to top-k
        if not(len(pred["prediction"]) == len(pred["evidence"]) == len(pred["title"]) == args.psg_top_k):
            assert len(pred["prediction"]) == len(pred["evidence"]) == len(pred["title"]) < args.psg_top_k, \
                (len(pred["prediction"]), len(pred["evidence"]), len(pred["title"]))
            print(len(pred["prediction"]), len(pred["evidence"]), len(pred["title"]))

            pred["evidence"] += [pred["evidence"][-1]] * (args.psg_top_k - len(pred["prediction"]))
            pred["title"] += [pred["title"][-1]] * (args.psg_top_k - len(pred["prediction"]))
            pred["se_pos"] += [pred["se_pos"][-1]] * (args.psg_top_k - len(pred["prediction"]))
            pred["prediction"] += [pred["prediction"][-1]] * (args.psg_top_k - len(pred["prediction"]))
            assert len(pred["prediction"]) == len(pred["evidence"]) == len(pred["title"]) == args.psg_top_k

        # Used for markers
        START = '<p_start>'
        END = '<p_end>'
        se_idxs = [[se_pos[0], max(se_pos[0], se_pos[1])] for se_pos in pred["se_pos"]]

        # Return sentence
        if args.return_sent:
            sents = [[(X.text, X[0].idx) for X in nlp(evidence).sents] for evidence in pred['evidence']]
            sent_idxs = [
                sorted(set([sum(np.array([st[1] for st in sent]) <= se_idx[0]) - 1] + [sum(np.array([st[1] for st in sent]) <= se_idx[1]-1) - 1]))
                for se_idx, sent in zip(se_idxs, sents)
            ]
            se_idxs = [[se_pos[0]-sent[sent_idx[0]][1], se_pos[1]-sent[sent_idx[0]][1]] for se_pos, sent_idx, sent in zip(se_idxs, sent_idxs, sents)]
            if not all(pred.replace(' ', '') in ' '.join([sent[sidx][0] for sidx in range(sent_idx[0], sent_idx[-1]+1)]).replace(' ', '')
                       for pred, sent, sent_idx in zip(pred['prediction'], sents, sent_idxs)):
                import pdb; pdb.set_trace()
                pass

            # get sentence based on the window
            max_context_len = args.max_context_len - 2 if args.mark_phrase else args.max_context_len
            my_dict["ctxs"] = [
                # {"title": title, "text": ' '.join(' '.join([sent[sidx][0] for sidx in range(sent_idx[0], sent_idx[-1]+1)]).split()[:max_context_len])} 
                {"title": title, "text": ' '.join(' '.join([sent[sidx][0] for sidx in range(
                    max(0, sent_idx[0]-args.sent_window), min(sent_idx[-1]+1+args.sent_window, len(sent)))]
                    ).split()[:max_context_len])
                } 
                for title, sent, sent_idx in zip(pred["title"], sents, sent_idxs)
            ]
        # Return passagae
        else:
            my_dict["ctxs"] = [
                {"title": title, "text": ' '.join(evd.split()[:args.max_context_len])}
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
        assert all(len(ctx['text'].split()) <= args.max_context_len for ctx in my_dict["ctxs"])

    print(f"avg ctx len={sum(avg_len)/len(avg_len):.2f} for {len(my_pred)} preds")

    out_file = os.path.join(
        args.model_dir, 'pred',
        os.path.splitext(args.pred_file)[0] + 
        f'_{"sent" if args.return_sent else "psg"}-top{args.psg_top_k}{"_mark" if args.mark_phrase else ""}.json'
    )
    print(f"dump to {out_file}")
    json.dump(my_target, open(out_file, 'w'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--pred_file', type=str, default='')
    parser.add_argument('--psg_top_k', type=int, default=100)
    parser.add_argument('--max_context_len', type=int, default=999999999)
    parser.add_argument('--mark_phrase', default=False, action='store_true')
    parser.add_argument('--return_sent', default=False, action='store_true')
    parser.add_argument('--sent_window', type=int, default=0)
    args = parser.parse_args()

    main(args)
