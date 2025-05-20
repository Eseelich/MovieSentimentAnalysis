import re
import string
import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# 1. Text cleaning
stop_words = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    """
    Clean and tokenize input text using regex and stopword removal.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"/?u/\w+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = re.findall(r"\b[a-z]+\b", text)
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

# 2. Load and prepare labeled data (Excel)
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v:k for k,v in label2id.items()}

def load_labeled_data(file_map):
    dfs = []
    for movie_id, path in file_map.items():
        df = pd.read_excel(path, engine='openpyxl')
        df["movie"] = movie_id
        df["cleaned_body"] = df["comment_body"].apply(clean_text)
        df["label"] = df["sentiment"].map(label2id)
        dfs.append(df[["movie", "cleaned_body", "label"]])
    return pd.concat(dfs, ignore_index=True)

# 3. Stratified split & oversampling
def split_and_oversample(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        df, test_size=test_size,
        stratify=df[["movie","label"]],
        random_state=random_state
    )
    max_count = train_df["label"].value_counts().max()
    oversampled = [grp.sample(n=max_count, replace=True, random_state=random_state)
                   for _, grp in train_df.groupby("label")]
    train_bal = pd.concat(oversampled).sample(frac=1, random_state=random_state)
    return train_bal, test_df

# 4. Transformer Dataset
class SentimentDataset(Dataset):
    """
    PyTorch Dataset for Transformer models
    """
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
        item = {key: val.squeeze(0) for key, val in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 5. Train Transformer model Train Transformer model
def train_transformer(train_df, eval_df,
                      model_name="distilbert-base-uncased",
                      output_dir="sentiment_model",
                      epochs=5, lr=2e-5,
                      batch_size=8, eval_batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = SentimentDataset(train_df["cleaned_body"], train_df["label"], tokenizer)
    eval_ds = SentimentDataset(eval_df["cleaned_body"], eval_df["label"], tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    args = TrainingArguments(
        output_dir=output_dir, overwrite_output_dir=True,
        num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=0.01, logging_steps=50,
        eval_strategy="steps", eval_steps=200,
        save_strategy="steps", save_steps=200,
        save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="f1_macro"
    )
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        return {"accuracy": accuracy_score(labels,preds),
                "f1_macro": f1_score(labels,preds,average="macro")}
    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds, eval_dataset=eval_ds,
                      tokenizer=tokenizer, compute_metrics=compute_metrics)
    trainer.train()
    return trainer, tokenizer, model

# 6. Tune threshold
def tune_negative_threshold(trainer, thresholds=np.linspace(0.1,0.5,17)):
    out = trainer.predict(trainer.eval_dataset)
    probs = softmax(out.predictions,axis=1); labels=out.label_ids
    best={'f1':-1,'neg_thr':None}
    for thr in thresholds:
        preds=[0 if p[0]>=thr else 1+np.argmax(p[1:]) for p in probs]
        f1=f1_score(labels,preds,average="macro",labels=[0,1,2])
        if f1>best['f1']: best={'f1':f1,'neg_thr':thr}
    return best['neg_thr']

# 7. Train LinearSVC (with calibration)
def train_linear_svc(train_df, eval_df):
    vec = TfidfVectorizer(max_features=10000)
    X_train = vec.fit_transform(train_df['cleaned_body'])
    X_eval = vec.transform(eval_df['cleaned_body'])
    svc = LinearSVC()
    clf = CalibratedClassifierCV(svc)
    clf.fit(X_train, train_df['label'])
    # can evaluate calibration on eval_df if needed
    return vec, clf

# 8. Ensemble prediction
def ensemble_predict(transformer_model, tokenizer, svc_clf, vec,
                     texts, neg_thr, weight_tf=0.5):
    # transformer probs
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = transformer_model(**enc).logits.cpu().numpy()
    probs_tf = softmax(logits,axis=1)
    # SVC probs
    X = vec.transform(texts)
    probs_svc = svc_clf.predict_proba(X)
    # weighted average
    probs = weight_tf*probs_tf + (1-weight_tf)*probs_svc
    # apply threshold rule for final
    preds = [0 if p[0]>=neg_thr else 1+np.argmax(p[1:]) for p in probs]
    return preds