import math
from collections import Counter
from multiprocessing import Pool, cpu_count
import numpy as np

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned. 
"""


class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer
        self._inverted_index = {}

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        inverted_index = {}
        doc_lengths = []

        for doc_idx, document in enumerate(corpus):
            doc_len = len(document)
            doc_lengths.append(doc_len)
            num_doc += doc_len

            frequencies = Counter(document)
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                nd[word] = nd.get(word, 0) + 1
                if word not in inverted_index:
                    inverted_index[word] = ([], [])
                inverted_index[word][0].append(doc_idx)
                inverted_index[word][1].append(freq)

            self.corpus_size += 1

        self.doc_len = np.array(doc_lengths, dtype=np.float64)
        self.avgdl = num_doc / self.corpus_size if self.corpus_size > 0 else 0

        self._inverted_index = {
            word: (
                np.array(docs, dtype=np.int32),
                np.array(freqs, dtype=np.float64),
            )
            for word, (docs, freqs) in inverted_index.items()
        }
        return nd

    def _tokenize_corpus(self, corpus):
        workers = max(1, cpu_count() - 2)
        with Pool(workers) as pool:
            tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):
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
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
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
        self.average_idf = idf_sum / len(self.idf) if self.idf else 0

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """The ATIRE BM25 variant uses an idf function which uses a log(idf)

        score. To prevent negative idf scores, this algorithm also adds a floor
        to the idf value of epsilon. See [Trotman, A., X. Jia, M. Crane, Towards
        an Efficient and Effective Search Engine] for more info :param query:
        :return:
        """
        score = np.zeros(self.corpus_size, dtype=np.float64)
        for q, q_count in Counter(query).items():
            idf = self.idf.get(q)
            if not idf or q not in self._inverted_index:
                continue
            doc_ids, q_freq = self._inverted_index[q]
            doc_len = self.doc_len[doc_ids]
            score[doc_ids] += (
                q_count
                * idf
                * (
                    q_freq
                    * (self.k1 + 1)
                    / (
                        q_freq
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
        for q, q_count in Counter(query).items():
            idf = self.idf.get(q)
            if not idf:
                continue
            q_freq = np.fromiter(
                (self.doc_freqs[di].get(q, 0) for di in doc_ids),
                dtype=np.float64,
                count=len(doc_ids),
            )
            nz = np.nonzero(q_freq)[0]
            if len(nz) == 0:
                continue
            q_freq_nz = q_freq[nz]
            doc_len_nz = doc_len[nz]
            score[nz] += (
                q_count
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

    def get_scores(self, query):
        score = np.zeros(self.corpus_size, dtype=np.float64)
        for q, q_count in Counter(query).items():
            idf = self.idf.get(q)
            if not idf:
                continue
            base_score = idf * (self.k1 + 1) * self.delta / (self.k1 + self.delta)
            score += q_count * base_score
            if q in self._inverted_index:
                doc_ids, q_freq = self._inverted_index[q]
                doc_len = self.doc_len[doc_ids]
                ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
                val_sparse = (
                    idf
                    * (self.k1 + 1)
                    * (ctd + self.delta)
                    / (self.k1 + ctd + self.delta)
                )
                score[doc_ids] += q_count * (val_sparse - base_score)
        return score

    def get_batch_scores(self, query, doc_ids):
        """Calculate bm25 scores between query and subset of all docs."""
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids), dtype=np.float64)
        doc_len = self.doc_len[doc_ids]
        for q, q_count in Counter(query).items():
            idf = self.idf.get(q)
            if not idf:
                continue
            base_score = idf * (self.k1 + 1) * self.delta / (self.k1 + self.delta)
            score += q_count * base_score
            q_freq = np.fromiter(
                (self.doc_freqs[di].get(q, 0) for di in doc_ids),
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
            score[nz] += q_count * (val_nz - base_score)
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

    def get_scores(self, query):
        score = np.zeros(self.corpus_size, dtype=np.float64)
        for q, q_count in Counter(query).items():
            idf = self.idf.get(q)
            if not idf:
                continue
            score += q_count * idf * self.delta
            if q in self._inverted_index:
                doc_ids, q_freq = self._inverted_index[q]
                doc_len = self.doc_len[doc_ids]
                score[doc_ids] += (
                    q_count
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
        for q, q_count in Counter(query).items():
            idf = self.idf.get(q)
            if not idf:
                continue
            score += q_count * idf * self.delta
            q_freq = np.fromiter(
                (self.doc_freqs[di].get(q, 0) for di in doc_ids),
                dtype=np.float64,
                count=len(doc_ids),
            )
            nz = np.nonzero(q_freq)[0]
            if len(nz) == 0:
                continue
            score[nz] += (
                q_count
                * idf
                * (q_freq[nz] * (self.k1 + 1))
                / (
                    self.k1 * (1 - self.b + self.b * doc_len[nz] / self.avgdl)
                    + q_freq[nz]
                )
            )
        return score.tolist()
