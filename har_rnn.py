#!/usr/bin/env python3
"""
Human Activity Recognition (HAR) with Recurrent Neural Networks
================================================================
One file, one command:   python har_rnn.py

What it does
  1. Downloads the UCI HAR smartphone-sensor dataset (falls back to synthetic data if offline)
  2. Trains a stacked LSTM (optionally also GRU + SimpleRNN with --compare)
  3. Evaluates on the official test split
  4. Saves graphs (PNG) and an auto-built PowerPoint deck with YOUR real results

Outputs go to ./outputs :
  class_distribution.png, sample_signals.png, training_curves.png,
  confusion_matrix.png, per_class_f1.png, model_comparison.png (with --compare),
  results.json, har_lstm.keras, HAR_RNN_Presentation.pptx

Usage
  pip install -r requirements.txt
  python har_rnn.py                       # LSTM, 30 epochs
  python har_rnn.py --compare             # also trains GRU and SimpleRNN
  python har_rnn.py --epochs 10           # quick run
  python har_rnn.py --synthetic           # skip download, use synthetic data
"""
import argparse
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split

SEED = 42
URL = ("https://archive.ics.uci.edu/static/public/240/"
       "human+activity+recognition+using+smartphones.zip")
SIGNALS = ["body_acc_x", "body_acc_y", "body_acc_z",
           "body_gyro_x", "body_gyro_y", "body_gyro_z",
           "total_acc_x", "total_acc_y", "total_acc_z"]
LABELS = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
PALETTE = ["#1A9E8F", "#2E86DE", "#F39C12", "#E74C3C", "#8E44AD", "#34495E"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _find_root(base: Path):
    for p in base.rglob("Inertial Signals"):
        return p.parent.parent
    return None


def download_dataset(data_dir: Path) -> Path:
    root = _find_root(data_dir) if data_dir.exists() else None
    if root:
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    zpath = data_dir / "uci_har.zip"
    print("Downloading UCI HAR dataset (~60 MB) ...")
    urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(data_dir)
    for inner in data_dir.glob("*.zip"):          # the archive contains a nested zip
        if inner.name != "uci_har.zip":
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(data_dir)
    root = _find_root(data_dir)
    if root is None:
        raise FileNotFoundError("Could not locate 'Inertial Signals' after extraction.")
    return root


def _load_split(root: Path, split: str):
    chans = [pd.read_csv(root / split / "Inertial Signals" / f"{s}_{split}.txt",
                         sep=r"\s+", header=None).values for s in SIGNALS]
    X = np.dstack(chans).astype("float32")                         # (N, 128, 9)
    y = pd.read_csv(root / split / f"y_{split}.txt", header=None).values.ravel() - 1
    return X, y.astype("int64")


def load_uci_har(data_dir: Path):
    root = download_dataset(data_dir)
    Xtr, ytr = _load_split(root, "train")
    Xte, yte = _load_split(root, "test")
    return Xtr, ytr, Xte, yte


def make_synthetic(n=3000, seed=SEED):
    """Offline fallback: physically-inspired fake accelerometer/gyro windows."""
    rng = np.random.default_rng(seed)
    t = np.arange(128) / 50.0
    freq = [2.0, 1.6, 2.6, 0, 0, 0]
    grav = np.array([[1, .1, .1], [1, .1, .1], [1, .1, .1],
                     [.7, .5, .3], [1, 0, .05], [.1, .1, 1]])
    y = rng.integers(0, 6, n)
    X = np.zeros((n, 128, 9), "float32")
    for i, c in enumerate(y):
        dyn = freq[c] > 0
        amp = rng.uniform(.3, .6) if dyn else .02
        ph = rng.uniform(0, 6.28, 3)
        wave = np.stack([np.sin(2 * np.pi * freq[c] * t + p) for p in ph], 1) * amp
        body = wave + rng.normal(0, .03, (128, 3))
        X[i, :, 0:3] = body
        X[i, :, 3:6] = (np.cos(2 * np.pi * max(freq[c], .1) * t[:, None] + ph) * amp * 2
                        + rng.normal(0, .05, (128, 3)))
        X[i, :, 6:9] = grav[c] + body
    return X, y


def _split_tuple(X, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=seed)
    return Xtr, ytr, Xte, yte


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(kind, timesteps, n_feat, n_cls):
    from tensorflow.keras import layers, models, optimizers
    cell = {"LSTM": layers.LSTM, "GRU": layers.GRU, "SimpleRNN": layers.SimpleRNN}[kind]
    model = models.Sequential([
        layers.Input(shape=(timesteps, n_feat)),
        cell(64, return_sequences=True),
        layers.Dropout(0.3),
        cell(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_cls, activation="softmax"),
    ], name=f"HAR_{kind}")
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(kind, Xtr, ytr, Xval, yval, epochs, batch):
    from tensorflow.keras import callbacks
    model = build_model(kind, Xtr.shape[1], Xtr.shape[2], len(LABELS))
    cbs = [callbacks.EarlyStopping(patience=6, restore_best_weights=True),
           callbacks.ReduceLROnPlateau(factor=.5, patience=3, min_lr=1e-5)]
    print(f"\n=== Training {kind} ===")
    hist = model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=epochs,
                     batch_size=batch, callbacks=cbs, verbose=2)
    return model, hist.history


# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #
def plot_class_distribution(y, out):
    counts = np.bincount(y, minlength=len(LABELS))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(LABELS, counts, color=PALETTE)
    ax.bar_label(bars, padding=3)
    ax.set(title="Training set: samples per activity", ylabel="Windows")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=20)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def plot_sample_signals(X, y, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for c, ax in enumerate(axes.ravel()):
        idx = np.where(y == c)[0][0]
        for k, name in zip(range(6, 9), "xyz"):
            ax.plot(X[idx, :, k], label=f"acc {name}", lw=1.2)
        ax.set_title(LABELS[c], color=PALETTE[c], fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Total body acceleration (x, y, z): one 2.56 s window per activity")
    fig.supxlabel("Time step (50 Hz)")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def plot_training_curves(hist, out):
    ep = range(1, len(hist["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(ep, hist[key], label="train", color=PALETTE[1], lw=2)
        ax.plot(ep, hist["val_" + key], label="validation", color=PALETTE[2], lw=2)
        ax.set(title=f"Model {title}", xlabel="Epoch", ylabel=title)
        ax.legend(); ax.grid(alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def plot_confusion(cm, out):
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", xticklabels=LABELS,
                yticklabels=LABELS, cbar_kws={"label": "Row-normalised"}, ax=ax)
    ax.set(title="Confusion matrix (test set, counts)", xlabel="Predicted", ylabel="True")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def plot_per_class_f1(report, out):
    f1 = [report[l]["f1-score"] for l in LABELS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(LABELS[::-1], f1[::-1], color=PALETTE[::-1])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(xlim=(0, 1.08), title="Per-class F1-score (test set)", xlabel="F1")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def plot_model_comparison(scores, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(list(scores), list(scores.values()), color=PALETTE[:len(scores)])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(ylim=(0, 1.08), ylabel="Test accuracy", title="RNN variants compared")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


# --------------------------------------------------------------------------- #
# PowerPoint
# --------------------------------------------------------------------------- #
def build_presentation(res, out_dir: Path):
    from PIL import Image
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt

    NAVY, TEAL = RGBColor(0x0F, 0x24, 0x3E), RGBColor(0x1A, 0x9E, 0x8F)
    GREY, WHITE = RGBColor(0x44, 0x4F, 0x5C), RGBColor(0xFF, 0xFF, 0xFF)
    CARD = RGBColor(0xEE, 0xF4, 0xF7)

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    n = [0]

    def rect(s, x, y, w, h, color, shape=MSO_SHAPE.RECTANGLE):
        r = s.shapes.add_shape(shape, x, y, w, h)
        r.fill.solid(); r.fill.fore_color.rgb = color; r.line.fill.background()
        return r

    def text(s, t, x, y, w, h, size=18, bold=False, color=NAVY, align=PP_ALIGN.LEFT):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.alignment = align
        r = p.add_run(); r.text = t
        r.font.size, r.font.bold, r.font.color.rgb = Pt(size), bold, color

    def bullets(s, items, x, y, w, h, size=18):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame; tf.word_wrap = True
        for i, it in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(10)
            r = p.add_run(); r.text = "\u2022  " + it
            r.font.size, r.font.color.rgb = Pt(size), GREY

    def image(s, path, x, y, w, h):
        iw, ih = Image.open(path).size
        k = min(w / iw, h / ih)
        pw, ph = int(iw * k), int(ih * k)
        s.shapes.add_picture(str(path), x + (w - pw) // 2, y + (h - ph) // 2, pw, ph)

    def slide(title):
        s = prs.slides.add_slide(blank); n[0] += 1
        rect(s, 0, 0, prs.slide_width, Inches(1.1), NAVY)
        rect(s, 0, Inches(1.1), prs.slide_width, Inches(0.07), TEAL)
        text(s, title, Inches(0.6), Inches(0.2), Inches(12), Inches(0.8), 30, True, WHITE)
        text(s, str(n[0]), Inches(12.3), Inches(6.95), Inches(0.7), Inches(0.4), 12,
             color=GREY, align=PP_ALIGN.RIGHT)
        return s

    def card(s, x, y, value, label):
        rect(s, x, y, Inches(2.9), Inches(1.5), CARD, MSO_SHAPE.ROUNDED_RECTANGLE)
        text(s, value, x, y + Inches(0.15), Inches(2.9), Inches(0.8), 34, True, TEAL, PP_ALIGN.CENTER)
        text(s, label, x, y + Inches(0.95), Inches(2.9), Inches(0.5), 14, False, GREY, PP_ALIGN.CENTER)

    O = lambda f: out_dir / f
    cm = np.array(res["confusion_matrix"])
    off = cm.copy(); np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    f1s = res["per_class_f1"]
    best_c, worst_c = max(f1s, key=f1s.get), min(f1s, key=f1s.get)

    # 1 Title
    s = prs.slides.add_slide(blank); n[0] += 1
    rect(s, 0, 0, prs.slide_width, prs.slide_height, NAVY)
    rect(s, Inches(0.8), Inches(3.55), Inches(1.6), Inches(0.08), TEAL)
    text(s, "Human Activity Recognition\nusing Recurrent Neural Networks", Inches(0.8), Inches(1.5),
         Inches(11.5), Inches(2), 42, True, WHITE)
    text(s, "Classifying smartphone sensor signals with LSTM  |  UCI HAR dataset",
         Inches(0.8), Inches(3.85), Inches(11), Inches(0.6), 20, False, RGBColor(0xBF, 0xD4, 0xE0))

    # 2 Problem
    s = slide("Problem & objective")
    bullets(s, ["Goal: automatically recognise what a person is doing from wearable sensor data",
                "Applications: fitness tracking, elderly fall monitoring, health care, smart homes",
                "Input: raw accelerometer + gyroscope time series from a waist-worn smartphone",
                "Output: one of 6 activities: walking, upstairs, downstairs, sitting, standing, laying",
                "Why RNN? Sensor streams are sequences, and LSTM cells model temporal dependencies"],
            Inches(0.8), Inches(1.7), Inches(11.7), Inches(5), 22)

    # 3 Dataset
    s = slide("Dataset: UCI HAR" + ("  (synthetic stand-in)" if res["synthetic"] else ""))
    bullets(s, [f"{res['n_train']} training and {res['n_test']} test windows",
                "Each window: 128 time steps (2.56 s at 50 Hz)",
                "9 channels: body acc, body gyro, total acc (x, y, z)",
                "30 volunteers; official subject-wise train/test split" if not res["synthetic"] else "Synthetic offline stand-in data",
                "Six activity classes, roughly balanced"],
            Inches(0.6), Inches(1.7), Inches(5.4), Inches(5), 19)
    image(s, O("class_distribution.png"), Inches(6.2), Inches(1.5), Inches(6.7), Inches(5.3))

    # 4 Signals
    s = slide("What the sensors see")
    image(s, O("sample_signals.png"), Inches(0.5), Inches(1.4), Inches(12.3), Inches(5.5))

    # 5 Model
    s = slide("Model & training setup")
    bullets(s, ["Input (128 x 9), standardised per channel",
                "LSTM(64, return sequences) -> Dropout 0.3",
                "LSTM(64) -> Dropout 0.3",
                "Dense(64, ReLU) -> Dense(6, softmax)",
                f"{res['params']:,} trainable parameters"],
            Inches(0.8), Inches(1.7), Inches(6), Inches(5), 21)
    bullets(s, ["Optimiser: Adam, learning rate 1e-3",
                "Loss: sparse categorical cross-entropy",
                f"Batch size {res['batch']}, up to {res['epochs']} epochs",
                "Early stopping + LR reduction on plateau",
                "15% of training data held out for validation"],
            Inches(7), Inches(1.7), Inches(5.8), Inches(5), 21)

    # 6 Curves
    s = slide("Training behaviour")
    image(s, O("training_curves.png"), Inches(0.5), Inches(1.4), Inches(12.3), Inches(4.0))
    card(s, Inches(0.9), Inches(5.5), f"{res['test_accuracy']*100:.2f}%", "Test accuracy")
    card(s, Inches(4.2), Inches(5.5), f"{res['macro_f1']:.3f}", "Macro F1-score")
    card(s, Inches(7.5), Inches(5.5), str(res["epochs_run"]), "Epochs trained")
    card(s, Inches(10.4), Inches(5.5), f"{res['best_val_accuracy']*100:.1f}%", "Best val. accuracy")

    # 7 Confusion
    s = slide("Where does the model get confused?")
    image(s, O("confusion_matrix.png"), Inches(0.4), Inches(1.35), Inches(6.9), Inches(5.7))
    bullets(s, [f"Largest error: {LABELS[i]} predicted as {LABELS[j]} ({off[i, j]} windows)",
                f"Best class: {best_c} (F1 {f1s[best_c]:.3f})",
                f"Hardest class: {worst_c} (F1 {f1s[worst_c]:.3f})",
                "Similar static postures (e.g. sitting vs standing) are a typical source of confusion"],
            Inches(7.6), Inches(1.8), Inches(5.3), Inches(5), 19)

    # 8 F1
    s = slide("Per-class performance")
    image(s, O("per_class_f1.png"), Inches(0.5), Inches(1.4), Inches(7.2), Inches(5.5))
    bullets(s, [f"Accuracy {res['test_accuracy']:.3f}",
                f"Macro F1 {res['macro_f1']:.3f}",
                f"Weighted F1 {res['weighted_f1']:.3f}"],
            Inches(8.2), Inches(2.2), Inches(4.6), Inches(3), 22)

    # 9 Comparison
    if res.get("comparison"):
        s = slide("LSTM vs GRU vs SimpleRNN")
        image(s, O("model_comparison.png"), Inches(0.5), Inches(1.4), Inches(7.4), Inches(5.5))
        best = max(res["comparison"], key=res["comparison"].get)
        bullets(s, [f"{k}: {v*100:.2f}%" for k, v in res["comparison"].items()] +
                [f"Best variant: {best}"],
                Inches(8.3), Inches(2.0), Inches(4.6), Inches(4), 20)

    # 10 Conclusion
    s = slide("Conclusions & future work")
    bullets(s, [f"A stacked LSTM reaches {res['test_accuracy']*100:.1f}% test accuracy on 6 activities",
                "Raw sensor windows are enough; no hand-crafted features required",
                f"Hardest class: {worst_c}; similar postures are easy to mix up",
                "Next: CNN-LSTM hybrids, attention, data augmentation, on-device TFLite deployment",
                "Next: subject-independent evaluation and more activities"],
            Inches(0.8), Inches(1.7), Inches(11.7), Inches(5), 22)

    path = out_dir / "HAR_RNN_Presentation.pptx"
    prs.save(path)
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def evaluate(model, Xte, yte):
    pred = model.predict(Xte, verbose=0).argmax(1)
    return pred, accuracy_score(yte, pred)


def main():
    ap = argparse.ArgumentParser(description="HAR with RNNs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--compare", action="store_true", help="also train GRU and SimpleRNN")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (no download)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-ppt", action="store_true")
    a = ap.parse_args()

    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    random.seed(SEED); np.random.seed(SEED)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    synthetic = a.synthetic
    if not synthetic:
        try:
            Xtr, ytr, Xte, yte = load_uci_har(Path(a.data_dir))
        except Exception as e:                                   # offline etc.
            print(f"[warn] Could not load UCI HAR ({e}).\n       Falling back to SYNTHETIC data.")
            synthetic = True
    if synthetic:
        Xtr, ytr, Xte, yte = _split_tuple(*make_synthetic(), SEED)
    print(f"Train {Xtr.shape}  Test {Xte.shape}  (synthetic={synthetic})")

    # per-channel standardisation using training statistics only
    mu, sd = Xtr.mean((0, 1), keepdims=True), Xtr.std((0, 1), keepdims=True) + 1e-8
    Xtr_n, Xte_n = (Xtr - mu) / sd, (Xte - mu) / sd
    Xt, Xv, yt, yv = train_test_split(Xtr_n, ytr, test_size=.15, stratify=ytr, random_state=SEED)

    plot_class_distribution(ytr, out / "class_distribution.png")
    plot_sample_signals(Xtr, ytr, out / "sample_signals.png")

    kinds = ["LSTM"] + (["GRU", "SimpleRNN"] if a.compare else [])
    comparison, main_model, main_hist = {}, None, None
    for k in kinds:
        model, hist = train_model(k, Xt, yt, Xv, yv, a.epochs, a.batch)
        _, acc = evaluate(model, Xte_n, yte)
        comparison[k] = float(acc)
        if k == "LSTM":
            main_model, main_hist = model, hist
    model = main_model
    model.summary()

    pred, acc = evaluate(model, Xte_n, yte)
    cm = confusion_matrix(yte, pred)
    rep = classification_report(yte, pred, target_names=LABELS, output_dict=True, digits=4)
    print("\n" + classification_report(yte, pred, target_names=LABELS, digits=4))

    plot_training_curves(main_hist, out / "training_curves.png")
    plot_confusion(cm, out / "confusion_matrix.png")
    plot_per_class_f1(rep, out / "per_class_f1.png")
    if a.compare:
        plot_model_comparison(comparison, out / "model_comparison.png")

    res = dict(synthetic=synthetic, n_train=int(len(ytr)), n_test=int(len(yte)),
               params=int(model.count_params()), batch=a.batch, epochs=a.epochs,
               epochs_run=len(main_hist["loss"]), test_accuracy=float(acc),
               macro_f1=float(f1_score(yte, pred, average="macro")),
               weighted_f1=float(f1_score(yte, pred, average="weighted")),
               best_val_accuracy=float(max(main_hist["val_accuracy"])),
               confusion_matrix=cm.tolist(),
               per_class_f1={l: float(rep[l]["f1-score"]) for l in LABELS}, comparison=comparison if a.compare else {})
    (out / "results.json").write_text(json.dumps(res, indent=2))
    model.save(out / "har_lstm.keras")

    if not a.no_ppt:
        print("PowerPoint:", build_presentation(res, out))
    print(f"\nDone. Test accuracy: {acc:.4f}.  Everything is in ./{out}/")


if __name__ == "__main__":
    main()
