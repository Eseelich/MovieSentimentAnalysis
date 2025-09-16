# -*- coding: utf-8 -*-
"""
Fine-tune DistilBERT for 3-class sentiment classification
– with threshold‐based inference to recover the negative class
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
import numpy as np
from scipy.special import softmax
from tqdm import tqdm


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# 1) Load and prepare your labeled data
df = pd.read_excel("sample_labeled.xlsx", engine="openpyxl")
label2id = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["sentiment"].map(label2id)

# 2) Split train/test stratified
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

# 3) Oversample minority classes
max_count = train_df["label"].value_counts().max()
oversampled = []
for lbl, grp in train_df.groupby("label"):
    oversampled.append(grp.sample(n=max_count, replace=True, random_state=42))
train_bal = pd.concat(oversampled).sample(frac=1, random_state=42)

# 4) Prepare tokenizer and datasets
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_ds = SentimentDataset(train_bal["comment_body"], train_bal["label"], tokenizer)
eval_ds  = SentimentDataset(test_df["comment_body"],    test_df["label"],    tokenizer)

# 5) Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)

# 6) TrainingArguments (match eval/save)
training_args = TrainingArguments(
    output_dir="hf-sentiment-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=20,

    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,

    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
)

# 7) Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# 8) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 9) Train & evaluate
trainer.train()
metrics = trainer.evaluate()
print("Eval metrics:", metrics)

# 10) Get raw logits on the eval set
pred_output = trainer.predict(eval_ds)
logits = pred_output.predictions  # shape (N,3)
probs  = softmax(logits, axis=1)

# 11) Tune thresholds on eval set
neg_thrs = np.linspace(0.1, 0.5, 9)
pos_thrs = np.linspace(0.1, 0.5, 9)
best = {"f1": -1, "neg_thr": 0.0, "pos_thr": 0.0}

true_labels = pred_output.label_ids
for neg in neg_thrs:
    for pos in pos_thrs:
        # assign labels by thresholds
        pred_labels = []
        for p in probs:
            if p[0] >= neg:
                pred_labels.append(0)
            elif p[2] >= pos:
                pred_labels.append(2)
            else:
                pred_labels.append(1)
        f1 = f1_score(true_labels, pred_labels, average="macro", labels=[0,1,2])
        if f1 > best["f1"]:
            best.update({"f1": f1, "neg_thr": neg, "pos_thr": pos})

print(f"\nBest thresholds on dev: neg ≥ {best['neg_thr']}, pos ≥ {best['pos_thr']}, F1-macro={best['f1']:.4f}")

# 12) Show evaluation with tuned thresholds
pred_labels = []
for p in probs:
    if p[0] >= best["neg_thr"]:
        pred_labels.append(0)
    elif p[2] >= best["pos_thr"]:
        pred_labels.append(2)
    else:
        pred_labels.append(1)

print("\nTuned Classification Report:")
print(classification_report(
    true_labels, pred_labels,
    target_names=["negative","neutral","positive"]
))
print("Tuned Confusion Matrix:")
print(confusion_matrix(true_labels, pred_labels, labels=[0,1,2]))

# 13) Save the tuned model
trainer.save_model("distilbert-sentiment-final")
tokenizer.save_pretrained("distilbert-sentiment-final")

# 14) Inference on full dataset with thresholds
df_full = pd.read_csv("Pirates 2 .csv")
# tokenize full corpus
enc = tokenizer(
    df_full["comment_body"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
with torch.no_grad():
    logits_full = model(**enc).logits.cpu().numpy()
probs_full = softmax(logits_full, axis=1)

# apply tuned thresholds
def apply_thresh(p):
    if p[0] >= best["neg_thr"]:
        return "negative"
    if p[2] >= best["pos_thr"]:
        return "positive"
    return "neutral"

df_full["sentiment"] = [apply_thresh(p) for p in probs_full]
df_full.to_csv("Pirates_with_custom_sentiment.csv", index=False)
print("\nDone – predictions with tuned thresholds saved.")
