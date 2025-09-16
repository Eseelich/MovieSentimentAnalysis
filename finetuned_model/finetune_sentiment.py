# -*- coding: utf-8 -*-
"""
Fine-tune DistilBERT for 3-class sentiment classification
(using legacy TrainingArguments with matching eval/save strategies)
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from tqdm import tqdm


class SentimentDataset(Dataset):
    """PyTorch dataset wrapping text + label for Transformers."""
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
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# 1) Load your labeled data
#    Make sure sample_labeled.xlsx has columns: comment_body, sentiment
df = pd.read_excel('sample_labeled.xlsx', engine='openpyxl')
label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['sentiment'].map(label2id)

# 2) Stratified train/test split
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)

# 3) Tokenizer + Datasets
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = SentimentDataset(train_df['comment_body'], train_df['label'], tokenizer)
eval_dataset  = SentimentDataset(test_df['comment_body'],  test_df['label'],  tokenizer)

# 4) Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)

# 5) TrainingArguments with matching eval_strategy & save_strategy
training_args = TrainingArguments(
    output_dir='hf-sentiment-finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    logging_dir='logs',
    logging_steps=10,

    do_train=True,
    do_eval=True,
    eval_strategy='steps',    # run evaluation every `eval_steps`
    eval_steps=100,
    save_strategy='steps',    # save checkpoints every `save_steps`
    save_steps=100,
    save_total_limit=1,       # keep only the last checkpoint

    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True
)

# 6) Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro')
    }

# 7) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 8) Train
trainer.train()
#%%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# 1) Run prediction to get raw logits + true labels
pred_output = trainer.predict(eval_dataset)

# 2) Turn logits into class‐indices
preds = np.argmax(pred_output.predictions, axis=1)
labels = pred_output.label_ids

# 3) Overall accuracy
acc = accuracy_score(labels, preds)
print(f"Test Accuracy: {acc:.4f}\n")

# 4) Per‐class precision/recall/f1
print("Classification Report:")
print(classification_report(
    labels, preds,
    target_names=['negative','neutral','positive']
))

# 5) Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(labels, preds, labels=[0,1,2]))

# 10) Save fine-tuned model
trainer.save_model('distilbert-sentiment-final')

# 11) Inference on full dataset
df_full = pd.read_csv('Pirates 2 .csv')
sentiment_pipe = pipeline(
    'sentiment-analysis',
    model='distilbert-sentiment-final',
    tokenizer=model_name
)
tqdm.pandas(desc="Predicting full data")
df_full['sentiment'] = df_full['comment_body'].progress_apply(
    lambda txt: sentiment_pipe(txt, truncation=True)[0]['label'].lower()
)

# 12) Save predictions
df_full.to_csv('Pirates_with_custom_sentiment.csv', index=False)
print("Done: Pirates_with_custom_sentiment.csv created.")
