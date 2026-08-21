"""
End-to-end training + evaluation script for the SMS Spam Naive Bayes model.

Run with:  python src/train.py
"""
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from naive_bayes import clean_and_tokenize, NaiveBayesTextClassifier

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spam.csv")


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, encoding="latin-1")
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]]
    elif {"Category", "Message"}.issubset(df.columns):
        df = df[["Category", "Message"]]
    else:
        raise KeyError(f"Unrecognized columns: {list(df.columns)}")
    df.columns = ["label", "text"]
    df["label_num"] = (df["label"] == "spam").astype(int)
    return df


def main():
    df = load_data()
    print(f"Loaded {len(df)} messages "
          f"({(df['label_num']==0).sum()} ham / {(df['label_num']==1).sum()} spam)")

    # Single, one-time split. Test set is untouched until final evaluation.
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label_num"], random_state=42
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    train_tokens = [clean_and_tokenize(t) for t in train_df["text"]]
    test_tokens = [clean_and_tokenize(t) for t in test_df["text"]]

    # Sanity check: no empty vocab, no fully-empty tokenized doc list
    non_empty = sum(1 for t in train_tokens if len(t) > 0)
    assert non_empty > 0, "Feature engineering produced zero usable training documents!"

    # Hyperparameter search on an internal train/validation split (test untouched)
    tr_tok, val_tok, tr_lab, val_lab = train_test_split(
        train_tokens, list(train_df["label_num"]), test_size=0.2,
        stratify=train_df["label_num"], random_state=42
    )

    best_alpha, best_f1 = None, -1
    print("\nHyperparameter search (alpha):")
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        model = NaiveBayesTextClassifier(alpha=alpha).fit(tr_tok, tr_lab)
        preds = model.predict(val_tok)
        f1 = f1_score(val_lab, preds, pos_label=1)
        print(f"  alpha={alpha:>4}  F1(val)={f1:.4f}")
        if f1 > best_f1:
            best_alpha, best_f1 = alpha, f1
    print(f"Selected alpha={best_alpha} (F1(val)={best_f1:.4f})")

    # Final training on the FULL training set
    final_model = NaiveBayesTextClassifier(alpha=best_alpha).fit(
        train_tokens, list(train_df["label_num"])
    )

    # Training sanity checks
    assert final_model.vocab_size_ > 0, "Vocabulary is empty after training!"
    assert abs(sum(final_model.class_priors_.values()) - 1.0) < 1e-9, "Priors do not sum to 1!"
    print(f"\nVocabulary size: {final_model.vocab_size_}")
    print(f"Class priors: {final_model.class_priors_}")

    # Evaluation on test set (touched for the first and only time here)
    test_preds = final_model.predict(test_tokens)
    f1 = f1_score(test_df["label_num"], test_preds, pos_label=1)
    acc = accuracy_score(test_df["label_num"], test_preds)
    prec = precision_score(test_df["label_num"], test_preds, pos_label=1)
    rec = recall_score(test_df["label_num"], test_preds, pos_label=1)
    cm = confusion_matrix(test_df["label_num"], test_preds)

    print("\n=== Test set results ===")
    print(f"F1 (spam):        {f1:.4f}")
    print(f"Precision (spam):  {prec:.4f}")
    print(f"Recall (spam):     {rec:.4f}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Confusion matrix:\n{cm}")

    # Regression guardrails: fail loudly if quality drops far below expectation
    assert f1 > 0.80, f"F1 dropped unexpectedly low ({f1:.4f}) - investigate before submitting!"

    print("\nTraining and evaluation completed successfully.")
    return {"f1": f1, "precision": prec, "recall": rec, "accuracy": acc, "alpha": best_alpha}


if __name__ == "__main__":
    main()
