# -*- coding: utf-8 -*-
"""
Created on Mon May 19 12:56:17 2025

@author: soras
"""

# -*- coding: utf-8 -*-
"""
VADER-Based Sentiment Analysis and Evaluation

This script:
1) Loads your manually labeled data (sample_labeled.xlsx)
2) Uses VADER to assign sentiment labels
3) Compares VADER’s predictions against your true labels
4) Applies VADER to the full Reddit comments dataset and exports results
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# 1) Load your labeled test set
#    Make sure 'sample_labeled.xlsx' has columns: comment_body, sentiment
df_test = pd.read_excel('sample_labeled.xlsx', engine='openpyxl')

# 2) Initialize the VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# 3) Define a helper to map compound scores to three classes
def vader_label(text, pos_thr=0.05, neg_thr=-0.05):
    score = analyzer.polarity_scores(text)['compound']
    if score >= pos_thr:
        return 'positive'
    elif score <= neg_thr:
        return 'negative'
    else:
        return 'neutral'

# 4) Annotate the test set
tqdm.pandas(desc="Annotating test data")
df_test['pred_sentiment'] = df_test['comment_body'].progress_apply(vader_label)

# 5) Evaluate VADER on your labeled data
y_true = df_test['sentiment']
y_pred = df_test['pred_sentiment']

print(f"Test Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(
    y_true, y_pred,
    target_names=['negative', 'neutral', 'positive']
))
print("\nConfusion Matrix:")
print(confusion_matrix(
    y_true, y_pred,
    labels=['negative', 'neutral', 'positive']
))

# 6) Apply VADER to the full comments dataset
df_full = pd.read_csv('Pirates 2 .csv')  # adjust path if needed
tqdm.pandas(desc="Annotating full dataset")
df_full['sentiment'] = df_full['comment_body'].progress_apply(vader_label)

# 7) Save the results
df_full.to_csv('Pirates_with_vader_sentiment.csv', index=False)
print("\nDone: 'Pirates_with_vader_sentiment.csv' created.")
