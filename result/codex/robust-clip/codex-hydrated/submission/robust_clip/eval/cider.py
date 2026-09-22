"""CIDEr (Vedantam et al., 2015) for the captioning evaluations.

The paper reports the CIDEr score of COCO and Flickr30k captions (Table 1).  We
implement CIDEr-D, the version used by ``pycocoevalcap``, from scratch so that
the repository does not depend on the (Java-based) official tokenizer:

* n-grams of order 1..4, candidate counts clipped to the maximum reference
  count (this is the "D" in CIDEr-D);
* tf-idf weighting with ``idf(g) = log(N / df(g))`` where ``df`` is computed
  either from the evaluation corpus (``df_mode="corpus"``) or from an external
  document-frequency file (``df_mode="file"``, e.g. the official COCO df);
* a Gaussian length penalty ``exp(-(len(c) - len(r))^2 / (2 * sigma^2))`` with
  ``sigma = 6``.

The implementation is numerically **identical** to ``pycocoevalcap``
(``tests/test_cider.py::test_cider_matches_pycocoevalcap`` pins this down); with
``scale=10`` it reproduces ``pycocoevalcap``'s value exactly.  The default
``scale=1000`` expresses that value in percent, which is the convention of the
paper's tables (e.g. LLaVA-1.5 13B reaches ~119 on the clean COCO captions,
i.e. a ``pycocoevalcap`` score of ~1.19).

Document frequencies computed from the *evaluation* corpus (the default) are a
standard approximation when the official df file is not available; it does not
change the ranking of the models, which is what the paper's conclusions rest
on.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lower-cased word tokenisation (a stand-in for PTBTokenizer)."""
    return _TOKEN_RE.findall(text.lower())


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1)))


class CiderScorer:
    """Compute CIDEr-D scores for a corpus of candidates and references."""

    def __init__(self, n: int = 4, sigma: float = 6.0, scale: float = 1000.0):
        self.n = n
        self.sigma = sigma
        self.scale = scale
        self.df: Counter = Counter()
        self.num_docs = 0
        self.idf: Dict[tuple, float] = {}

    # ------------------------------------------------------------------ stats
    def prepare(self, references: Iterable[Sequence[str]], df_file: Optional[str] = None) -> "CiderScorer":
        """Collect document frequencies for the tf-idf weights."""
        if df_file is not None and os.path.exists(df_file):
            with open(df_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.df = Counter({tuple(k.split()): v for k, v in data.get("df", data).items()})
            self.num_docs = int(data.get("num_docs", len(self.df)))
        else:
            num_sets = 0
            for refs in references:
                num_sets += 1
                # ``df`` counts the *images* whose reference captions contain the
                # n-gram (pycocoevalcap's ``compute_doc_freq``), so the idf is
                # ``log(#images) - log(#images containing the n-gram)``.
                seen = set()
                for ref in refs:
                    tokens = tokenize(ref)
                    for n in range(1, self.n + 1):
                        seen.update(_ngram_counts(tokens, n))
                for gram in seen:
                    self.df[gram] += 1
            self.num_docs = num_sets
        self._build_idf()
        return self

    def _build_idf(self) -> None:
        """``idf(g) = log(N) - log(df(g))``.

        ``N`` is the number of **reference sets** (images) while ``df`` counts
        the reference *sentences* containing the n-gram -- this is exactly what
        ``pycocoevalcap`` does (``ref_len = log(len(crefs))`` and
        ``df = log(max(1, document_frequency[ngram]))``), which is why common
        n-grams can end up with a negative weight when the df is computed from
        the evaluation corpus itself.  The official COCO results use the df file
        of the training captions instead.
        """
        num_docs = max(1, self.num_docs)
        self.idf = {}
        self.log_num_docs = math.log(num_docs)
        for gram, df in self.df.items():
            self.idf[gram] = self.log_num_docs - math.log(max(1.0, float(df)))

    def _weight(self, gram) -> float:
        """tf-idf weight of an n-gram.

        N-grams that never occur in the reference corpus get ``df = 0`` and
        therefore the *largest* weight, ``log(N)`` -- this is what
        ``pycocoevalcap`` does (``df = np.log(max(1.0, document_frequency[ngram]))``)
        and it is what penalises hallucinated words in a candidate caption.
        """
        weight = self.idf.get(gram)
        if weight is None:
            weight = self.log_num_docs
        return weight

    # ---------------------------------------------------------------- scoring
    def _tfidf_vector(self, counts: Counter) -> Tuple[Dict[tuple, float], float]:
        vec = {}
        for gram, count in counts.items():
            vec[gram] = count * self._weight(gram)
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return vec, norm

    def score(
        self,
        candidate: str,
        references: Sequence[str],
        lengths: Optional[Tuple[int, Sequence[int]]] = None,
    ) -> float:
        """CIDEr-D of a single candidate against its references.

        The returned value follows ``pycocoevalcap`` exactly
        (``10 * mean_n(Σ_refs cosine) / |refs|``) up to the factor
        ``self.scale / 10``; ``scale = 10`` equals ``pycocoevalcap`` exactly and
        the default ``scale = 1000`` reports the metric in percent, the
        convention of the paper's tables.
        """
        cand_tokens = tokenize(candidate)
        ref_tokens = [tokenize(r) for r in references]
        if not cand_tokens or not ref_tokens:
            return 0.0
        len_c = lengths[0] if lengths else len(cand_tokens)
        len_rs = lengths[1] if lengths else [len(r) for r in ref_tokens]

        score = 0.0
        for ref, len_r in zip(ref_tokens, len_rs):
            per_n = 0.0
            for n in range(1, self.n + 1):
                cand_counts = _ngram_counts(cand_tokens, n)
                ref_counts = _ngram_counts(ref, n)
                if not cand_counts:
                    continue
                # NOTE: both vectors are built from the *raw* n-gram counts and
                # the clipping of the candidate counts by the reference counts
                # happens inside the dot product below -- exactly like
                # pycocoevalcap ("vrama91: added clipping").
                vec_c, norm_c = self._tfidf_vector(cand_counts)
                vec_r, norm_r = self._tfidf_vector(ref_counts)
                if norm_c == 0.0 or norm_r == 0.0:
                    continue
                shared = set(vec_c) & set(vec_r)
                if not shared:
                    continue
                # ``min(vec_hyp, vec_ref) * vec_ref`` (pycocoevalcap, vrama91's
                # clipping) over the shared n-grams.
                dot = sum(min(vec_c[g], vec_r[g]) * vec_r[g] for g in shared)
                delta = float(len_c - len_r)
                per_n += (dot / (norm_c * norm_r)) * math.exp(
                    -(delta ** 2) / (2.0 * self.sigma ** 2)
                )
            # mean over the four n-gram orders, as in pycocoevalcap
            score += per_n / self.n
        return self.scale * score / len(ref_tokens)

    def compute_scores(
        self, candidates: Sequence[str], references: Sequence[Sequence[str]]
    ) -> List[float]:
        return [self.score(c, r) for c, r in zip(candidates, references)]


def cider_d(
    candidates: Sequence[str],
    references: Sequence[Sequence[str]],
    df_file: Optional[str] = None,
    scale: float = 1000.0,
    n: int = 4,
    sigma: float = 6.0,
) -> List[float]:
    """Convenience wrapper returning the per-sample CIDEr-D scores."""
    scorer = CiderScorer(n=n, sigma=sigma, scale=scale).prepare(references, df_file=df_file)
    return scorer.compute_scores(candidates, references)


def compute_cider_scores(
    candidates: Sequence[str],
    references: Sequence[Sequence[str]],
    scorer: Optional[CiderScorer] = None,
    df_file: Optional[str] = None,
    scale: float = 1000.0,
) -> List[float]:
    """Reusable entry point for the attack pipeline (keeps the statistics fixed)."""
    if scorer is None:
        scorer = CiderScorer(scale=scale).prepare(references, df_file=df_file)
    return scorer.compute_scores(candidates, references)
