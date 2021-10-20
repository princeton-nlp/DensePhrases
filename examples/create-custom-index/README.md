# Creating a Custom Phrase Index with DensePhrases

Basically, DensePhrases uses a text corpus pre-processed in the following format (a snippet from [articles.json](https://github.com/princeton-nlp/DensePhrases/blob/main/examples/create-custom-index/articles.json)):
```
{
    "data": [
        {
            "title": "America's Got Talent (season 4)",
            "paragraphs": [
                {
                    "context": " The fourth season of \"America's Got Talent\", ... Country singer Kevin Skinner was named the winner on September 16, 2009 ..."
                },
                {
                    "context": " Season four was Hasselhoff's final season as a judge. This season started broadcasting live on August 4, 2009. ..."
                },
                ...
            ]
        },
    ]
}
```

## Building a Phrase Index
Each `context` contains a single natural paragraph of a variable length. The following command creates phrase vectors for the custom corpus (`articles.json`) with the `densephrases-multi` model.

```bash
python generate_phrase_vecs.py \
    --model_type bert \
    --pretrained_name_or_path SpanBERT/spanbert-base-cased \
    --data_dir ./ \
    --cache_dir $CACHE_DIR \
    --predict_file examples/create-custom-index/articles.json \
    --do_dump \
    --max_seq_length 512 \
    --doc_stride 500 \
    --fp16 \
    --filter_threshold -2.0 \
    --append_title \
    --load_dir $SAVE_DIR/densephrases-multi \
    --output_dir $SAVE_DIR/densephrases-multi_sample
```
The phrase vectors (and their metadata) will be saved under `$SAVE_DIR/densephrases-multi_sample/dump/phrase`. Now you need to create a faiss index as follows:
```bash
python build_phrase_index.py \
    --dump_dir $SAVE_DIR/densephrases-multi_sample/dump \
    --stage all \
    --replace \
    --num_clusters 32 \
    --fine_quant OPQ96 \
    --doc_sample_ratio 1.0 \
    --vec_sample_ratio 1.0 \
    --cuda

# Compress metadata for faster inference
python scripts/preprocess/compress_metadata.py \
    --input_dump_dir $SAVE_DIR/densephrases-multi_sample/dump/phrase \
    --output_dir $SAVE_DIR/densephrases-multi_sample/dump
```
Note that this example uses a very small text corpus and the hyperparameters for `build_phrase_index.py` in a larger scale corpus can be found [here](https://github.com/princeton-nlp/DensePhrases/tree/main#densephrases-training-indexing-and-inference).
Depending on the size of the corpus, the hyperparameters should change as follows:
* `num_clusters`: Set to make the number of vectors per cluster < 2000 (e.g., `--num_culsters 256` works well for `dev_wiki.json`).
* `doc/vec_sample_ratio`: Use the default value (0.2) except for the small scale experiments (shown above).
* `fine_quant`: Currently only OPQ96 is supported.

The phrase index (with IVFOPQ) will be saved under `$SAVE_DIR/densephrases-multi_sample/dump/start`.
For creating a large-scale phrase index (e.g., Wikipedia), see [dump_phrases.py](https://github.com/princeton-nlp/DensePhrases/blob/main/scripts/parallel/dump_phrases.py) for an example, which is also explained [here](https://github.com/princeton-nlp/DensePhrases/tree/main#2-creating-a-phrase-index).

## Testing a Phrase Index
You can use this phrase index to run a [demo](https://github.com/princeton-nlp/DensePhrases/tree/main#playing-with-a-densephrases-demo) or evaluate your set of queries.
For instance, you can feed a set of questions (`questions.json`) to the custom phrase index as follows:
```bash
python eval_phrase_retrieval.py \
    --run_mode eval \
    --cuda \
    --dump_dir $SAVE_DIR/densephrases-multi_sample/dump \
    --index_name start/32_flat_OPQ96 \
    --load_dir $SAVE_DIR/densephrases-multi \
    --test_path examples/create-custom-index/questions.json \
    --save_pred \
    --truecase
```
The prediction file will be saved as `$SAVE_DIR/densephrases-multi/pred/questions_3_top10.pred`, which shows the answer phrases and the passages that contain the phrases:
```
{
    "1": {
        "question": "Who won season 4 of America's got talent",
        ...
        "prediction": [
            "Kevin Skinner",
            ...
        ],
        "evidence": [
            "The fourth season of \"America's Got Talent\", an American television reality show talent competition, premiered on the NBC network on June 23, 2009. Country singer Kevin Skinner was named the winner on September 16, 2009.",
            ...
        ],
    }
    ...
}
```
