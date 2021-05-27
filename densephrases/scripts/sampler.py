import glob
import heapq
import json
import functools
import operator
from dataclasses import dataclass
import shutil
from hashlib import sha224
from pathlib import Path
from typing import Iterable, List, Set, Union

import numpy as np
from joblib import delayed, Parallel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi


@dataclass
class Article:
    file: str
    file_idx: int
    article_idx: int

    def __hash__(self) -> int:
        return hash((self.file_idx, self.article_idx))

    def __lt__(self, other: "Article") -> bool:
        if self.file_idx == other.file_idx:
            return self.article_idx < other.article_idx
        return self.file_idx < other.file_idx


@dataclass
class Paragraph:
    article: Article
    paragraph_idx: int

    def __hash__(self) -> int:
        return hash((self.article, self.paragraph_idx))

    def __lt__(self, other: "Paragraph") -> bool:
        return (
            self.paragraph_idx < other.paragraph_idx
            if self.article == other.article
            else self.article < other.article
        )


@dataclass
class QuestionMatches:
    question_indices: List[int]
    matches_question: List[Set[Paragraph]]
    matches_answers: List[Set[Paragraph]]
    matches_global: Set[Paragraph]
    matches_global_articles: Set[Article]

    @property
    def paragraph_cnt(self) -> int:
        return len(self.matches_global)

    @property
    def article_cnt(self) -> int:
        return len(self.matches_global_articles)

    def __add__(self, other: "QuestionMatches") -> "QuestionMatches":
        return QuestionMatches(
            question_indices=self.question_indices + other.question_indices,
            matches_question=self.matches_question + other.matches_question,
            matches_answers=self.matches_answers + other.matches_answers,
            matches_global=self.matches_global.union(other.matches_global),
            matches_global_articles=self.matches_global_articles.union(
                other.matches_global_articles
            ),
        )

    @classmethod
    def empty(cls) -> "QuestionMatches":
        return QuestionMatches(
            question_indices=[],
            matches_question=[],
            matches_answers=[],
            matches_global=set(),
            matches_global_articles=set(),
        )

    @classmethod
    def single_query(
        cls,
        question_idx: int,
        matches_question: Set[Paragraph],
        matches_answers: Set[Paragraph],
    ) -> "QuestionMatches":
        matches_global = matches_question.union(matches_answers)
        return QuestionMatches(
            question_indices=[question_idx],
            matches_question=[matches_question],
            matches_answers=[matches_answers],
            matches_global=matches_global,
            matches_global_articles={match.article for match in matches_global},
        )


class BM25Sklearn(object):
    """
    Taken from https://gist.github.com/koreyou/f3a8a0470d32aa56b32f198f49a9f2b8
    """

    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """Fit IDF to documents X"""
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()
        self.len_y = y.sum(1).A1
        self.y_csc = y.tocsc()

    def transform(self, q, n: int):
        """Calculate BM25 between query q and documents X"""
        b, k1, avdl = self.b, self.k1, self.avdl
        (q,) = super(TfidfVectorizer, self.vectorizer).transform([q])
        X = self.y_csc[:, q.indices]
        den = X + (k1 * (1 - b + b * self.len_y / avdl))[:, None]
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.0
        num = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        res = (num / den).sum(1).A1
        return (np.argsort(res)[-n:])[::-1]


@dataclass
class QAWikiDumpSampler:
    """
    Goal is to sub-sample a QA dataset alongside the "relevant" wikipedia paragraphs.

    Steps:
    1- Build a BM25 index of the Wikipedia dump
    2- Retrieve top paragraphs for each question in the QA dataset
    3- Repeat 2 until the union of all retrieved paragraphs hits a target count.

    Note: for step 2 we'll actually retrieve the union of two queries:
    - `top_k_questions` paragraphs retrieved by querying the question
    - `top_k_answers` paragraphs retrieved by querying a space-separated concatenation of all the answers
      e.g. if answers are [Carl McCormick, Johnnie Newt] we'll query "Carl McCormick Johnnie Newt"

    pros: we'll ensure that our OpenQA system has a chance of finding some of the answers.
    cons: we may make it too easy for our system.

    we should ensure that top_k_questions >> top_k_answers so that we're not making retrieval too easy.
    """

    path_qa: str
    path_wiki: str
    path_cache: str
    path_cwd: Path = Path.cwd()
    top_k_questions: int = 200
    top_k_answers: int = 5
    target_paragraph_cnt: int = 1_000_000
    clear_cache: bool = False
    use_sklearn = True
    wiki_sample: float = 1
    random_state: int = 0
    n_jobs: int = -1
    index = None

    def __post_init__(self):
        path_cache = Path(self.path_cache)
        if self.clear_cache and path_cache.exists():
            shutil.rmtree(path_cache)
        path_cache.mkdir(exist_ok=True)
        print("reading QA dataset into a dataframe...")
        self.df_qa = self._read_qa_df()
        print(f"QA df has {len(self.df_qa)} rows")
        print("reading wiki dump into a dataframe...")
        self.df_wiki = self._read_wiki_df()
        print(f"wiki df has {len(self.df_wiki)} rows")
        print("run `build_index` next before running queries")

    def build_index(self):
        print("building BM25 index of the wiki paragraphs")
        self.index = self._build_index_wiki_paragraphs()
        print("done building the index. you can use the `query` method now.")

    def _query_top_indices(self, query: str, top_n: int) -> Iterable[int]:
        query_tokenized = self._tokenize(query)
        top = heapq.nlargest(
            top_n,
            (
                (score, idx)
                for idx, score in enumerate(self.index.get_scores(query_tokenized))
                if score > 0
            ),
        )
        return [x[1] for x in top]

    def _query_top_indices_sklearn(self, query: str, top_n: int) -> Iterable[int]:
        return self.index.transform(query, top_n)

    def query(
        self, query: str, top_n: int, cache: bool = True, return_df: bool = True
    ) -> Union[pd.DataFrame, Set[Paragraph]]:
        """
        By default it tries to locate the (query, top_n) in cache. If not available it'll compute and then cache.
        """
        key = self._hash_key(query, top_n)
        path = self._hash_key_path(key)
        if cache and path.exists():
            df_res = pd.read_csv(path)
        else:
            top_indices_fn = (
                self._query_top_indices_sklearn
                if self.use_sklearn
                else self._query_top_indices
            )
            top_indices = top_indices_fn(query, top_n)
            df_res = self.df_wiki.iloc[top_indices]
            if cache:
                df_res.to_csv(path, index=False)
        return df_res if return_df else self._extract_paragraphs_set(df_res)

    @property
    def path_cache_abs(self) -> Path:
        return Path(f"{self.path_cwd}/{self.path_cache}")

    def _hash_key_path(self, key: str) -> Path:
        return self.path_cache_abs / Path(f"{key}.csv")

    def _build_index_wiki_paragraphs(self) -> Union[BM25Okapi, BM25Sklearn]:
        paragraphs = self.df_wiki["paragraph"]
        return self._build_index(paragraphs)

    def _build_index(self, corpus: Iterable[str]) -> Union[BM25Okapi, BM25Sklearn]:
        if self.use_sklearn:
            bm25 = BM25Sklearn()
            bm25.fit(corpus)
            return bm25
        else:
            tokenized_corpus = [self._tokenize(doc) for doc in corpus]
            return BM25Okapi(tokenized_corpus)

    def _read_qa_df(self) -> pd.DataFrame:
        qa = self._load_json(self.path_qa)["data"]
        return pd.DataFrame(
            {
                "id": row["id"],
                "question": row["question"],
                "answer_cnt": len(row["answers"]),
                "answers": row["answers"],
            }
            for row in qa
        )

    def _read_wiki_df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "file": file,
                "file_idx": file_idx,
                "article_idx": article_idx,
                "paragraph_idx": paragraph_idx,
                "title": row["title"],
                "paragraph_cnt": len(row["paragraphs"]),
                "paragraph": par["context"],
                "paragraph_char_cnt": len(par["context"]),
            }
            for file_idx, file in enumerate(sorted(glob.glob(f"{self.path_wiki}/*")))
            for article_idx, row in enumerate(self._load_json(file)["data"])
            for paragraph_idx, par in enumerate(row["paragraphs"])
        )
        return (
            df.sample(
                n=int(len(df) * self.wiki_sample), random_state=self.random_state
            ).reset_index(drop=True)
            if self.wiki_sample < 1
            else df
        )

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        return s.lower().split(" ")

    @staticmethod
    def _hash_key(query: str, n: int) -> str:
        return sha224(bytearray(f"{query}_{n}", "utf8")).hexdigest()

    @staticmethod
    def _load_json(path) -> dict:
        return json.load(open(path))

    @staticmethod
    def _extract_paragraphs_set(df: pd.DataFrame) -> Set[Paragraph]:
        return {
            Paragraph(
                article=Article(
                    file=row["file"],
                    file_idx=row["file_idx"],
                    article_idx=row["article_idx"],
                ),
                paragraph_idx=row["paragraph_idx"],
            )
            for _, row in df.iterrows()
        }

    def _retrieve_for_single_query(
        self, question_idx: int, question: str, answers: List[str], cache: bool = True
    ) -> QuestionMatches:
        answers_str = " ".join(answers)
        return QuestionMatches.single_query(
            question_idx=question_idx,
            matches_question=self.query(
                question, top_n=self.top_k_questions, cache=cache, return_df=False
            ),
            matches_answers=self.query(
                answers_str, top_n=self.top_k_answers, cache=cache, return_df=False
            ),
        )

    def process_queries(
        self,
        n: int,
        cache: bool = True,
        n_jobs: int = 1,
        sharedmem=True,
    ) -> QuestionMatches:
        iterable = zip(
            range(n), self.df_qa["question"].iloc[:n], self.df_qa["answers"].iloc[:n]
        )
        init = QuestionMatches.empty()
        fn = self._retrieve_for_single_query
        par = Parallel(n_jobs=n_jobs, require="sharedmem" if sharedmem else None)
        results = par(delayed(fn)(*params, cache=cache) for params in iterable)
        return functools.reduce(operator.add, results, init)
