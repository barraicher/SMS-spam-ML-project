"""
Naive Bayes text classifier, implemented from scratch (no sklearn.naive_bayes).

This module contains:
  - clean_and_tokenize: simple text -> token-list feature engineering
  - NaiveBayesTextClassifier: Multinomial Naive Bayes with Laplace smoothing
"""
import re
import numpy as np

STOPWORDS = set("""a an the and or but if is are was were be been being to of in on for with
as at by from this that these those it its it's i you he she we they my your his her our their
me him them not no do does did so than then there here just so very can will would should could
u ur 2 4""".split())


def clean_and_tokenize(text: str) -> list[str]:
    """Lowercase, strip non-letters, tokenize, drop stopwords and 1-char tokens."""
    if not isinstance(text, str):
        raise TypeError(f"clean_and_tokenize expects a string, got {type(text)}")
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


class NaiveBayesTextClassifier:
    """
    Multinomial Naive Bayes for text classification, implemented from scratch.

    Parameters
    ----------
    alpha : float
        Laplace smoothing hyperparameter (must be > 0).
    """

    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha must be > 0 (Laplace smoothing requires alpha > 0)")
        self.alpha = alpha
        self.classes_ = None
        self.class_priors_ = None
        self.word_counts_ = None
        self.total_words_ = None
        self.vocab_ = None
        self.vocab_size_ = None

    def fit(self, tokenized_docs, labels):
        if len(tokenized_docs) != len(labels):
            raise ValueError("tokenized_docs and labels must have the same length")
        if len(labels) == 0:
            raise ValueError("Cannot fit on an empty dataset")

        self.classes_ = sorted(set(labels))
        n_docs = len(labels)

        self.class_priors_ = {
            c: sum(1 for l in labels if l == c) / n_docs for c in self.classes_
        }

        self.word_counts_ = {c: {} for c in self.classes_}
        self.total_words_ = {c: 0 for c in self.classes_}
        vocab = set()

        for tokens, label in zip(tokenized_docs, labels):
            for w in tokens:
                vocab.add(w)
                self.word_counts_[label][w] = self.word_counts_[label].get(w, 0) + 1
                self.total_words_[label] += 1

        self.vocab_ = vocab
        self.vocab_size_ = len(vocab)
        return self

    def _log_likelihood(self, word, c):
        count = self.word_counts_[c].get(word, 0)
        return np.log(
            (count + self.alpha) / (self.total_words_[c] + self.alpha * self.vocab_size_)
        )

    def predict_one(self, tokens):
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        scores = {}
        for c in self.classes_:
            score = np.log(self.class_priors_[c])
            for w in tokens:
                if w in self.vocab_:  # unseen word overall -> skip (no info)
                    score += self._log_likelihood(w, c)
            scores[c] = score
        return max(scores, key=scores.get)

    def predict(self, tokenized_docs):
        return [self.predict_one(t) for t in tokenized_docs]
