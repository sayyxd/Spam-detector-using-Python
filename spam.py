"""
Spam Message Detector - classify SMS as SPAM or NOT SPAM
Model: TF-IDF + Multinomial Naive Bayes (scikit-learn)

Setup:
    pip install scikit-learn pandas joblib

Dataset: UCI "SMS Spam Collection" (5,572 labelled messages).
    Run `python spam_detector.py train --download` to fetch a copy automatically,
    or supply your own file with --data (tab-separated "label<TAB>message",
    or a Kaggle-style spam.csv with the label in column 1 and text in column 2).

Usage:
    python spam_detector.py train --download
    python spam_detector.py train --data sms.tsv
    python spam_detector.py predict "WINNER!! Claim your free prize now"
    python spam_detector.py chat          # type messages one by one
"""
import argparse
import re
import sys
import urllib.request
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

MODEL_PATH = Path("spam_model.joblib")
DEFAULT_DATA = Path("sms.tsv")
DATA_URL = ("https://raw.githubusercontent.com/justmarkham/"
            "pycon-2016-tutorial/master/data/sms.tsv")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Lowercase and replace spam-typical patterns with placeholder tokens.

    Spam often hinges on URLs, phone numbers, prices and shouting, so instead of
    deleting them we keep a token that tells the model "a URL was here".
    """
    text = str(text)
    text = re.sub(r"(https?://\S+|www\.\S+)", " urltoken ", text, flags=re.I)
    text = re.sub(r"[£$€]\s?\d[\d,.]*|\d[\d,.]*\s?(?:p|gbp|usd|rs)\b", " moneytoken ", text, flags=re.I)
    text = re.sub(r"\b\d{5,}\b", " longnumbertoken ", text)   # phones, short codes
    text = re.sub(r"\b\d{1,4}\b", " numtoken ", text)
    if sum(c.isupper() for c in text) > 0.4 * max(len(text), 1):
        text += " shoutingtoken"
    return re.sub(r"\s+", " ", text.lower()).strip()


def download_data(dest: Path = DEFAULT_DATA) -> Path:
    print(f"Downloading dataset to {dest} ...")
    try:
        urllib.request.urlretrieve(DATA_URL, dest)
    except Exception as e:  # network blocked, URL moved, etc.
        sys.exit(f"Download failed ({e}). Get the SMS Spam Collection manually "
                 "(see top of file) and pass it with --data.")
    return dest


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="latin-1").iloc[:, :2]
    else:
        df = pd.read_csv(path, sep="\t", header=None, quoting=3, encoding="utf-8")
    df.columns = ["label", "message"]
    df["label"] = df["label"].astype(str).str.strip().str.lower().map({"spam": 1, "ham": 0})
    df = df.dropna().drop_duplicates(subset="message")
    df["label"] = df["label"].astype(int)
    df["content"] = df["message"].map(clean_text)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", MultinomialNB(alpha=0.1)),
    ])


def show_top_words(model: Pipeline, n: int = 12) -> None:
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    log_prob = model.named_steps["clf"].feature_log_prob_
    diff = log_prob[1] - log_prob[0]  # how much more likely in spam than ham
    order = diff.argsort()
    print(f"\nTop spam indicators : {', '.join(vocab[order[-n:]][::-1])}")
    print(f"Top normal indicators: {', '.join(vocab[order[:n]])}")


def train(path: Path) -> None:
    df = load_dataset(path)
    n_spam = int(df.label.sum())
    print(f"Loaded {len(df)} unique messages ({n_spam} spam, {len(df) - n_spam} not spam)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    model = build_pipeline().fit(X_train, y_train)
    preds = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Spam caught: {tp}/{tp + fn}   |   Good messages wrongly flagged: {fp}/{tn + fp}")
    print("\n" + classification_report(y_test, preds, target_names=["NOT SPAM", "SPAM"], zero_division=0))
    show_top_words(model)

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run: python spam_detector.py train --download")
    return joblib.load(MODEL_PATH)


def classify(model: Pipeline, message: str, threshold: float = 0.5) -> None:
    p_spam = model.predict_proba([clean_text(message)])[0][1]
    verdict = "SPAM" if p_spam >= threshold else "NOT SPAM"
    print(f"  -> {verdict}  (spam probability {p_spam:.1%})")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="SMS Spam Detector")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train and save a model")
    t.add_argument("--data", type=Path, default=DEFAULT_DATA)
    t.add_argument("--download", action="store_true", help="download the dataset first")

    p = sub.add_parser("predict", help="classify one message")
    p.add_argument("message")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="spam probability cutoff (raise it to flag less aggressively)")

    sub.add_parser("chat", help="classify messages interactively")

    args = parser.parse_args()

    if args.cmd == "train":
        if args.download or not args.data.exists():
            if not args.download:
                sys.exit(f"{args.data} not found. Use --download or --data <file>.")
            download_data(args.data)
        train(args.data)
    elif args.cmd == "predict":
        classify(load_model(), args.message, args.threshold)
    else:
        model = load_model()
        print("Type a message and press Enter (empty line to quit).")
        while True:
            msg = input("\nSMS> ").strip()
            if not msg:
                break
            classify(model, msg)


if __name__ == "__main__":
    main()
