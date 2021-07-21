## Create SQuAD-Style Wiki Dump (20181220)

### Download wiki dump of 20181220
```
python download_wikidump.py \
    --output_dir /hdd1/data/wikidump
```

### Extract Wiki dump via Wikiextractor
Use [Wikiextractor](https://github.com/attardi/wikiextractor) to convert wiki dump into the json style.

```
python WikiExtractor.py \
    --filter_disambig_pages \
    --json \
    -o /hdd1/data/wikidump/extracted/ \
    /hdd1/data/wikidump/enwiki-20181220-pages-articles.xml.bz2
```

### Build docs.db in SQlite style
```
python build_db.py \
    --data_path /hdd1/data/wikidump/extracted \
    --save_path /hdd1/data/wikidump/docs_20181220.db \
    --preprocess prep_wikipedia.py
```

### Transform sqlite to squad-style
```
python build_wikisquad.py \
    --db_path /hdd1/data/wikidump/docs_20181220.db \
    --out_dir /hdd1/data/wikidump/20181220
```

### Concatenate short length of paragraphs
```
python concat_wikisquad.py \
    --input_dir /hdd1/data/wikidump/20181220 \
    --output_dir /hdd1/data/wikidump/20181220_concat
```
