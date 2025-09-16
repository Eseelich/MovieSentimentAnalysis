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
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# Label mappings
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

# Precompute stopwords
stop_words = set(ENGLISH_STOP_WORDS)

# Text cleaning
def clean_text(text):
    # 0) Immer erst String draus machen
    text = str(text)

    # 1) Lowercase
    text = text.lower()
    # 2) Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # 3) Remove markdown links ([text](url))
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    # 4) Remove Reddit user references
    text = re.sub(r"/?u/\w+", "", text)
    # 5) Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # 6) Remove digits and punctuation
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 7) Tokenize on whitespace & non-word boundaries
    tokens = re.findall(r"\b[a-z]+\b", text)
    # 8) Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

def load_labeled_data(file_map):
    dfs = []
    for movie_id, path in file_map.items():
        df = pd.read_excel(path, engine="openpyxl")
        df = df.dropna(subset=["comment_body", "sentiment"])
        df["movie"] = movie_id
        df["cleaned_body"] = df["comment_body"].apply(clean_text)

        # Wert-weise Mapping-Funktion
        def map_label(x):
            # Falls schon int, direkt übernehmen
            if isinstance(x, (int, np.integer)):
                return int(x)
            # Sonst String → lowercase → trim → map
            xs = str(x).lower().strip()
            return label2id.get(xs, None)  # None, wenn unbekanntes Label

        df["label"] = df["sentiment"].apply(map_label)
        # Zeilen ohne gültiges Label entfernen
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        dfs.append(df[["movie", "cleaned_body", "label"]])

    return pd.concat(dfs, ignore_index=True)

# 3. Split & oversample
def split_and_oversample(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(  
        df, test_size=test_size,
        stratify=df[['movie', 'label']],
        random_state=random_state
    )
    max_count = train_df['label'].value_counts().max()
    oversampled = [grp.sample(n=max_count, replace=True, random_state=random_state)
                   for _, grp in train_df.groupby('label')]
    train_bal = pd.concat(oversampled).sample(frac=1, random_state=random_state)
    return train_bal, test_df

# 4. Transformer Dataset
torch.backends.cudnn.benchmark = True
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
            self.texts[idx], truncation=True,
            padding='max_length', max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 5. Train transformer (faster: fewer epochs, eval per epoch)
def train_transformer(train_df, eval_df,
                      model_name='distilbert-base-uncased',
                      epochs=3, lr=2e-5,
                      batch_size=16, eval_batch_size=32,
                      output_dir='tx_model'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = SentimentDataset(train_df['cleaned_body'], train_df['label'], tokenizer)
    eval_ds = SentimentDataset(eval_df['cleaned_body'], eval_df['label'], tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    args = TrainingArguments(
        output_dir=output_dir, overwrite_output_dir=True,
        num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=0.01,
        eval_strategy='epoch',  # renamed for compatibility
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro'
    )
    def compute_metrics(pred):
        logits, labels = pred
        preds = logits.argmax(axis=1)
        return {'accuracy': accuracy_score(labels, preds),
                'f1_macro': f1_score(labels, preds, average='macro')}
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        tokenizer=tokenizer, compute_metrics=compute_metrics
    )
    trainer.train()
    return trainer, tokenizer, model

# 6. Tune threshold (coarser grid)
def tune_negative_threshold(trainer,
                             thresholds=np.linspace(0.1, 0.5, 9)):
    out = trainer.predict(trainer.eval_dataset)
    probs = softmax(out.predictions, axis=1)
    labels = out.label_ids
    best = {'f1': -1, 'neg_thr': None}
    for thr in thresholds:
        preds = [0 if p[0] >= thr else 1 + np.argmax(p[1:]) for p in probs]
        f1_val = f1_score(labels, preds, average='macro', labels=[0,1,2])
        if f1_val > best['f1']:
            best = {'f1': f1_val, 'neg_thr': thr}
    return best['neg_thr']

# 7. Train classical models
def train_linear_svc(train_df, eval_df):
    vec = TfidfVectorizer(max_features=5000)
    X_train = vec.fit_transform(train_df['cleaned_body'])
    svc = LinearSVC()
    clf = CalibratedClassifierCV(svc)
    clf.fit(X_train, train_df['label'])
    return vec, clf

def train_logistic_regression(train_df, eval_df):
    vec = TfidfVectorizer(max_features=5000)
    lr = LogisticRegression(max_iter=500)
    clf = CalibratedClassifierCV(lr)
    clf.fit(vec.fit_transform(train_df['cleaned_body']), train_df['label'])
    return vec, clf

# 8. Ensemble predict
def ensemble_predict(models, vecs, texts, neg_thr, weights):
    tok, svc_vec, lr_vec = vecs
    tm, svc_clf, lr_clf = models
    # transformer probs
    enc = tok(texts, padding='max_length', truncation=True,
              max_length=128, return_tensors='pt')
    with torch.no_grad(): logits = tm(**enc).logits.cpu().numpy()
    probs_tf = softmax(logits, axis=1)
    # SVC probs
    probs_svc = svc_clf.predict_proba(svc_vec.transform(texts))
    # LR probs
    probs_lr = lr_clf.predict_proba(lr_vec.transform(texts))
    w_tf, w_svc, w_lr = weights
    probs = w_tf*probs_tf + w_svc*probs_svc + w_lr*probs_lr
    return [0 if p[0]>=neg_thr else 1+np.argmax(p[1:]) for p in probs]

