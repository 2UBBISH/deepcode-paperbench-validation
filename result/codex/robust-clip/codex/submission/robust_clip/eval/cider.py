"""CIDEr (Vedantam et al., 2015) -- the captioning metric of the paper.

The captioning evaluations of Sec. 4.1 report CIDEr, and App. B.6 explains how
the attack pipeline uses it: *"each [of the five ground truth captions] is
considered for computation of the CIDEr score [...] After each attack we compute
the CIDEr scores and do not attack the samples anymore that already have a score
below 10 or 2 for COCO and Flickr30k respectively."*

The implementation follows ``pycocoevalcap``, the reference implementation used
by the COCO captioning challenge (clipping of the n-gram counts, Gaussian length
penalty with ``sigma = 6``, corpus level document frequencies and the factor 10):

    IDF(ngram)     = log(#images) - log(max(1, #images containing the ngram))
    score          = 100 * 10 * mean_n [ sum_ngram min(tf_c, tf_r) * tfidf_r ]
                     / (||tfidf_c|| * ||tfidf_r||) * exp(-(len_c - len_r)^2 / (2 sigma^2))

The factor 100 is not part of the original metric but is the convention of the
paper, whose CIDEr values are reported as percentages (e.g. 79.7 for LLaVA with
the original CLIP on COCO, i.e. 0.797 in the standard scaling).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

__all__ = ["Cider", "tokenize"]

_PUNCT = re.compile(r"[^\w\s]")


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer after lower casing and removing punctuation.

    (The reference implementation uses the PTB tokenizer; for the captioning
    benchmarks of the paper the two agree on the reported trends.)
    """
    text = text.strip().lower()
    text = _PUNCT.sub(" ", text)
    return [token for token in text.split() if token]


def _count_ngrams(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i : i + n])] += 1
    return counts


class Cider:
    """Corpus level CIDEr scorer.

    Parameters
    ----------
    n:
        maximal n-gram order (4 in the paper).
    sigma:
        width of the length penalty (6.0, as in the reference implementation).
    scale:
        100 to report CIDEr in percent (the scale of the paper), 1 for the
        standard scaling of the original metric (where a perfect match scores 10).
    """

    def __init__(self, n: int = 4, sigma: float = 6.0, scale: float = 100.0):
        self.n = n
        self.sigma = sigma
        self.scale = scale
        self.df: Dict[Tuple[str, ...], int] = {}
        self.num_images: int = 0

    # ------------------------------------------------------------------
    # corpus level document frequencies
    # ------------------------------------------------------------------
    def fit(self, references: Sequence[Sequence[str]]) -> "Cider":
        """Compute the document frequencies on the reference corpus.

        As in the reference implementation the IDFs are *corpus level* statistics:
        they are computed once on all reference captions of the evaluation set
        (e.g. the 5 captions of every one of the 500 / 5000 evaluation images) and
        then reused for every candidate.
        """
        self.df = dict(self._document_frequency(references))
        self.num_images = len(references)
        return self

    # ------------------------------------------------------------------
    # corpus level statistics
    # ------------------------------------------------------------------
    def _document_frequency(self, references: Sequence[Sequence[str]]) -> Dict[Tuple[str, ...], int]:
        df: Dict[Tuple[str, ...], int] = defaultdict(int)
        for refs in references:
            seen = set()
            for ref in refs:
                tokens = tokenize(ref)
                for n in range(1, self.n + 1):
                    seen.update(_count_ngrams(tokens, n).keys())
            for ngram in seen:
                df[ngram] += 1
        return df

    def _vector(self, tokens: Sequence[str], df: Dict[Tuple[str, ...], int], ref_len: float):
        """tf-idf vectors of all n-gram orders + norms + sentence length."""
        vectors: List[Dict[Tuple[str, ...], float]] = [defaultdict(float) for _ in range(self.n)]
        norms = [0.0] * self.n
        length = 0
        for n in range(1, self.n + 1):
            counts = _count_ngrams(tokens, n)
            for ngram, term_freq in counts.items():
                idf = ref_len - math.log(max(1.0, float(df.get(ngram, 1))))
                weight = float(term_freq) * idf
                vectors[n - 1][ngram] = weight
                norms[n - 1] += weight * weight
                if n == 2:
                    # the reference implementation uses the number of *bigrams*
                    # as "length" of the sentence in the Gaussian length penalty
                    length += term_freq
        norms = [math.sqrt(value) for value in norms]
        return vectors, norms, length

    def _similarity(self, vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref) -> List[float]:
        delta = float(length_hyp - length_ref)
        values = [0.0] * self.n
        for n in range(self.n):
            value = 0.0
            for ngram, count in vec_hyp[n].items():
                value += min(count, vec_ref[n][ngram]) * vec_ref[n][ngram]
            if norm_hyp[n] != 0 and norm_ref[n] != 0:
                value /= norm_hyp[n] * norm_ref[n]
            values[n] = value * math.exp(-(delta ** 2) / (2 * self.sigma ** 2))
        return values

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def score(self, candidate: str, references: Sequence[str], df: Dict = None, num_images: int = None) -> float:
        """CIDEr of a single candidate against the references of *one* image."""
        references = list(references)
        if df is None:
            df = self.df if self.df else self._document_frequency([references])
        if num_images is None:
            num_images = self.num_images or 1
        ref_len = math.log(float(max(num_images, 1)))
        vec_hyp, norm_hyp, length_hyp = self._vector(tokenize(candidate), df, ref_len)
        score = 0.0
        for ref in references:
            vec_ref, norm_ref, length_ref = self._vector(tokenize(ref), df, ref_len)
            score += float(sum(self._similarity(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref)))
        if references:
            score /= len(references)
        # mean over the n-gram orders and the factor 10 of CIDEr (resp. CIDEr-D)
        return score / self.n * 10.0 * self.scale

    def compute_score(self, references: Dict[str, Sequence[str]], candidates: Dict[str, str], verbose: int = 0):
        """Corpus level CIDEr, ``pycocoevalcap``-style.

        ``references`` and ``candidates`` map an image id to the list of ground
        truth captions, respectively to the generated caption.
        """
        refs_list = [references[key] for key in candidates]
        self.fit(refs_list)
        scores = [self.score(candidates[key], references[key]) for key in candidates]
        mean = sum(scores) / max(len(scores), 1)
        return mean, scores

    def score_batch(
        self,
        candidates: Sequence[str],
        references: Sequence[Sequence[str]],
        use_fitted_df: bool = True,
    ) -> List[float]:
        """Per-sample CIDEr.

        If :meth:`fit` was called (recommended: on the full reference corpus) the
        corpus statistics are reused, otherwise they are estimated on the batch.
        """
        refs_list = [list(refs) for refs in references]
        df = self.df if (use_fitted_df and self.df) else self._document_frequency(refs_list)
        num_images = self.num_images if (use_fitted_df and self.num_images) else len(refs_list)
        return [self.score(cand, refs, df=df, num_images=num_images) for cand, refs in zip(candidates, refs_list)]
