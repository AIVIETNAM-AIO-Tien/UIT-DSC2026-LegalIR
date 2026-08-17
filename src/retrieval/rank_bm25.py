import math
from typing import Sequence, cast, overload
import numpy as np

from collections import Counter
from multiprocessing import Pool, cpu_count

from numpy._typing import NDArray

from src.types import CorpusTokens, RawText, Token, TokenizerFunc

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned. 
"""


type InvertedIndexTable = dict[
    Token, tuple[NDArray[np.int32], NDArray[np.float64]]
] 


class BM25:
    @overload
    def __init__(
        self,
        corpus: CorpusTokens,
        tokenizer: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        corpus: Sequence[RawText],
        tokenizer: TokenizerFunc,
    ) -> None: ...

    def __init__(self, corpus: CorpusTokens | Sequence[RawText], tokenizer: TokenizerFunc | None = None):
        self.corpus_size: int = 0
        self.avgdl: float = 0
        self.doc_freqs: list[dict[Token, int]] = []
        self.idf: dict[Token, float] = {}
        self.doc_len: NDArray[np.float64]
        self.tokenizer: TokenizerFunc | None = tokenizer
        self._inverted_index: InvertedIndexTable = {}

        if tokenizer is not None:
            corpus = self._tokenize_corpus(cast(Sequence[RawText], corpus))
        else:
            corpus = cast(CorpusTokens, corpus)

        nd: dict[Token, int] = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus: CorpusTokens) -> dict[Token, int]:
        nd: dict[Token, int] = {}  # word -> number of documents with word
        num_doc: int = 0
        inverted_index: dict[Token, tuple[list[int], list[int]]] = {}
        doc_lengths: list[int] = []

        doc_idx: int
        tokenized_document: list[Token]
        for doc_idx, tokenized_document in enumerate(corpus):
            doc_len: int = len(tokenized_document)
            doc_lengths.append(doc_len)
            num_doc += doc_len

            frequencies: dict[Token, int] = Counter(tokenized_document)
            self.doc_freqs.append(frequencies)

            token: Token
            freq: int
            for token, freq in frequencies.items():
                nd[token] = nd.get(token, 0) + 1
                if token not in inverted_index:
                    inverted_index[token] = ([], [])
                inverted_index[token][0].append(doc_idx)
                inverted_index[token][1].append(freq)

            self.corpus_size += 1

        self.doc_len: NDArray[np.float64] = np.array(doc_lengths, dtype=np.float64)
        self.avgdl: float = num_doc / self.corpus_size if self.corpus_size > 0 else 0

        self._inverted_index: InvertedIndexTable = {
            token: (
                np.array(docs, dtype=np.int32),
                np.array(freqs, dtype=np.float64),
            )
            for token, (docs, freqs) in inverted_index.items()
        }
        return nd

    # this duplicate stuff, we should just instastiate this class with
    # pre-tokenized corpora and nuke this thing out of existence
    def _tokenize_corpus(self, corpus: Sequence[RawText]) -> CorpusTokens:
        if self.tokenizer is None: return []

        workers = max(1, cpu_count() - 2)
        with Pool(workers) as pool:
            tokenized_corpus: CorpusTokens = pool.map(self.tokenizer, corpus)

        return tokenized_corpus

    def _calc_idf(self, nd: dict[Token, int]):
        raise NotImplementedError()

    def get_scores(self, query: list[Token]) -> NDArray[np.float64]:
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query: list[Token], documents, n: int = 5):
        assert (
            self.corpus_size == len(documents)
        ), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        if self.corpus_size <= n:
            top_n = np.argsort(scores)[::-1]
        else:
            top_n = np.argpartition(scores, -n)[-n:]
            top_n = top_n[np.argsort(scores[top_n])[::-1]]

        return [documents[i] for i in top_n]


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1: float =1.5, b: float =0.75, epsilon: float =0.25):
        self.k1: float = k1
        self.b: float = b
        self.epsilon: float = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """Calculates frequencies of terms in documents and in corpus.

        This algorithm sets a floor on the idf values to eps * average_idf
        """
        idf_sum = 0
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf: float = idf_sum / len(self.idf) if self.idf else 0

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: list[Token]) -> NDArray[np.float64]:
        """The ATIRE BM25 variant uses an idf function which uses a log(idf)

        score. To prevent negative idf scores, this algorithm also adds a floor
        to the idf value of epsilon. See [Trotman, A., X. Jia, M. Crane, Towards
        an Efficient and Effective Search Engine] for more info :param query:
        :return:
        """
        score = np.zeros(self.corpus_size, dtype=np.float64)
        for tok, tok_count in Counter(query).items():
            idf = self.idf.get(tok)
            if not idf or tok not in self._inverted_index:
                continue
            doc_ids, tok_freq = self._inverted_index[tok]
            doc_len = self.doc_len[doc_ids]
            score[doc_ids] += (
                tok_count
                * idf
                * (
                    tok_freq
                    * (self.k1 + 1)
                    / (
                        tok_freq
                        + self.k1
                        * (1 - self.b + self.b * doc_len / self.avgdl)
                    )
                )
            )
        return score

    def get_batch_scores(self, query, doc_ids):
        """Calculate bm25 scores between query and subset of all docs."""
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids), dtype=np.float64)
        doc_len = self.doc_len[doc_ids]
        for tok, tok_count in Counter(query).items():
            idf = self.idf.get(tok)
            if not idf:
                continue
            q_freq = np.fromiter(
                (self.doc_freqs[di].get(tok, 0) for di in doc_ids),
                dtype=np.float64,
                count=len(doc_ids),
            )
            nz = np.nonzero(q_freq)[0]
            if len(nz) == 0:
                continue
            q_freq_nz = q_freq[nz]
            doc_len_nz = doc_len[nz]
            score[nz] += (
                tok_count
                * idf
                * (
                    q_freq_nz
                    * (self.k1 + 1)
                    / (
                        q_freq_nz
                        + self.k1
                        * (1 - self.b + self.b * doc_len_nz / self.avgdl)
                    )
                )
            )
        return score.tolist()


class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log((self.corpus_size + 1) / (freq + 0.5))
            self.idf[word] = idf

    def get_scores(self, query: list[Token]) -> NDArray[np.float64]:
        score = np.zeros(self.corpus_size, dtype=np.float64)
        for tok, tok_count in Counter(query).items():
            idf = self.idf.get(tok)
            if not idf:
                continue
            base_score = idf * (self.k1 + 1) * self.delta / (self.k1 + self.delta)
            score += tok_count * base_score
            if tok in self._inverted_index:
                doc_ids, q_freq = self._inverted_index[tok]
                doc_len = self.doc_len[doc_ids]
                ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
                val_sparse = (
                    idf
                    * (self.k1 + 1)
                    * (ctd + self.delta)
                    / (self.k1 + ctd + self.delta)
                )
                score[doc_ids] += tok_count * (val_sparse - base_score)
        return score

    def get_batch_scores(self, query, doc_ids):
        """Calculate bm25 scores between query and subset of all docs."""
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids), dtype=np.float64)
        doc_len = self.doc_len[doc_ids]
        for tok, tok_count in Counter(query).items():
            idf = self.idf.get(tok)
            if not idf:
                continue
            base_score = idf * (self.k1 + 1) * self.delta / (self.k1 + self.delta)
            score += tok_count * base_score
            q_freq = np.fromiter(
                (self.doc_freqs[di].get(tok, 0) for di in doc_ids),
                dtype=np.float64,
                count=len(doc_ids),
            )
            nz = np.nonzero(q_freq)[0]
            if len(nz) == 0:
                continue
            ctd = q_freq[nz] / (1 - self.b + self.b * doc_len[nz] / self.avgdl)
            val_nz = (
                idf
                * (self.k1 + 1)
                * (ctd + self.delta)
                / (self.k1 + ctd + self.delta)
            )
            score[nz] += tok_count * (val_nz - base_score)
        return score.tolist()


class BM25Plus(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log((self.corpus_size + 1) / freq)
            self.idf[word] = idf

    def get_scores(self, query: list[Token]) -> NDArray[np.float64]:
        score = np.zeros(self.corpus_size, dtype=np.float64)
        for tok, tok_count in Counter(query).items():
            idf = self.idf.get(tok)
            if not idf:
                continue
            score += tok_count * idf * self.delta
            if tok in self._inverted_index:
                doc_ids, q_freq = self._inverted_index[tok]
                doc_len = self.doc_len[doc_ids]
                score[doc_ids] += (
                    tok_count
                    * idf
                    * (q_freq * (self.k1 + 1))
                    / (
                        self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                        + q_freq
                    )
                )
        return score

    def get_batch_scores(self, query, doc_ids):
        """Calculate bm25 scores between query and subset of all docs."""
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids), dtype=np.float64)
        doc_len = self.doc_len[doc_ids]
        for tok, tok_count in Counter(query).items():
            idf = self.idf.get(tok)
            if not idf:
                continue
            score += tok_count * idf * self.delta
            q_freq = np.fromiter(
                (self.doc_freqs[di].get(tok, 0) for di in doc_ids),
                dtype=np.float64,
                count=len(doc_ids),
            )
            nz = np.nonzero(q_freq)[0]
            if len(nz) == 0:
                continue
            score[nz] += (
                tok_count
                * idf
                * (q_freq[nz] * (self.k1 + 1))
                / (
                    self.k1 * (1 - self.b + self.b * doc_len[nz] / self.avgdl)
                    + q_freq[nz]
                )
            )
        return score.tolist()
