"""
Unit tests for src/naive_bayes.py

Run with:  pytest tests/ -v
"""
import math
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naive_bayes import clean_and_tokenize, NaiveBayesTextClassifier


# ---------- clean_and_tokenize ----------

def test_tokenize_lowercases():
    assert clean_and_tokenize("FREE Money") == ["free", "money"]

def test_tokenize_removes_punctuation_and_digits():
    tokens = clean_and_tokenize("Win $1000 now!!!")
    assert "1000" not in tokens
    assert "$" not in " ".join(tokens)
    assert "win" in tokens
    assert "now" in tokens

def test_tokenize_removes_stopwords():
    tokens = clean_and_tokenize("this is a free gift for you")
    for sw in ["this", "is", "a", "for", "you"]:
        assert sw not in tokens
    assert "free" in tokens
    assert "gift" in tokens

def test_tokenize_rejects_non_string():
    with pytest.raises(TypeError):
        clean_and_tokenize(12345)

def test_tokenize_empty_string_returns_empty_list():
    assert clean_and_tokenize("") == []

def test_tokenize_only_stopwords_returns_empty_list():
    assert clean_and_tokenize("is the and or") == []


# ---------- NaiveBayesTextClassifier: basic correctness ----------

def _toy_dataset():
    # A tiny, clearly-separable spam/ham toy set
    docs = [
        ["free", "money", "win", "prize"],       # spam
        ["free", "cash", "winner", "claim"],      # spam
        ["win", "lottery", "cash", "prize"],      # spam
        ["hello", "how", "are", "you"],           # ham
        ["meeting", "tomorrow", "morning"],       # ham
        ["are", "you", "free", "tomorrow"],       # ham (has "free" but is ham -> tests it's not just keyword matching)
    ]
    labels = ["spam", "spam", "spam", "ham", "ham", "ham"]
    return docs, labels

def test_fit_learns_correct_priors():
    docs, labels = _toy_dataset()
    model = NaiveBayesTextClassifier(alpha=1.0).fit(docs, labels)
    assert model.class_priors_["spam"] == pytest.approx(0.5)
    assert model.class_priors_["ham"] == pytest.approx(0.5)

def test_predict_obvious_spam():
    docs, labels = _toy_dataset()
    model = NaiveBayesTextClassifier(alpha=1.0).fit(docs, labels)
    pred = model.predict_one(["win", "cash", "prize", "claim"])
    assert pred == "spam"

def test_predict_obvious_ham():
    docs, labels = _toy_dataset()
    model = NaiveBayesTextClassifier(alpha=1.0).fit(docs, labels)
    pred = model.predict_one(["meeting", "morning", "hello"])
    assert pred == "ham"

def test_predict_handles_unseen_words_gracefully():
    """Words never seen in training should be skipped, not crash the model."""
    docs, labels = _toy_dataset()
    model = NaiveBayesTextClassifier(alpha=1.0).fit(docs, labels)
    # "xyzzyunknown" was never seen in training
    pred = model.predict_one(["xyzzyunknown", "win", "prize"])
    assert pred in ("spam", "ham")  # should not raise

def test_predict_empty_token_list_falls_back_to_prior():
    """An empty message should be classified by class priors alone."""
    docs, labels = _toy_dataset()
    model = NaiveBayesTextClassifier(alpha=1.0).fit(docs, labels)
    pred = model.predict_one([])
    # With a 50/50 prior in the toy set, either is plausible - just verify no crash
    assert pred in ("spam", "ham")

def test_predict_batch_matches_predict_one():
    docs, labels = _toy_dataset()
    model = NaiveBayesTextClassifier(alpha=1.0).fit(docs, labels)
    batch_preds = model.predict(docs)
    single_preds = [model.predict_one(d) for d in docs]
    assert batch_preds == single_preds


# ---------- Guardrails / error handling ----------

def test_alpha_must_be_positive():
    with pytest.raises(ValueError):
        NaiveBayesTextClassifier(alpha=0)
    with pytest.raises(ValueError):
        NaiveBayesTextClassifier(alpha=-1)

def test_predict_before_fit_raises():
    model = NaiveBayesTextClassifier(alpha=1.0)
    with pytest.raises(RuntimeError):
        model.predict_one(["free", "money"])

def test_fit_mismatched_lengths_raises():
    model = NaiveBayesTextClassifier(alpha=1.0)
    with pytest.raises(ValueError):
        model.fit([["a", "b"], ["c"]], ["spam"])  # 2 docs, 1 label

def test_fit_empty_dataset_raises():
    model = NaiveBayesTextClassifier(alpha=1.0)
    with pytest.raises(ValueError):
        model.fit([], [])


# ---------- Laplace smoothing sanity check ----------

def test_smoothing_prevents_zero_probability():
    """
    A word that appears in 'spam' but never in 'ham' should NOT give ham
    a probability of exactly zero (that's the whole point of Laplace smoothing).
    """
    docs, labels = _toy_dataset()
    model = NaiveBayesTextClassifier(alpha=1.0).fit(docs, labels)
    # "lottery" only appears in spam docs
    log_p_ham = model._log_likelihood("lottery", "ham")
    assert math.isfinite(log_p_ham)  # not -inf / NaN

def test_higher_alpha_smooths_more():
    """With higher alpha, the gap between seen/unseen word likelihoods should shrink."""
    docs, labels = _toy_dataset()
    low_alpha = NaiveBayesTextClassifier(alpha=0.01).fit(docs, labels)
    high_alpha = NaiveBayesTextClassifier(alpha=5.0).fit(docs, labels)

    gap_low = low_alpha._log_likelihood("win", "spam") - low_alpha._log_likelihood("meeting", "spam")
    gap_high = high_alpha._log_likelihood("win", "spam") - high_alpha._log_likelihood("meeting", "spam")
    assert abs(gap_high) < abs(gap_low)


# ---------- No data leakage: vocabulary should come from train only ----------

def test_vocab_built_only_from_training_data():
    docs, labels = _toy_dataset()
    model = NaiveBayesTextClassifier(alpha=1.0).fit(docs, labels)
    assert "supercalifragilistic" not in model.vocab_
