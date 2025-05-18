from transformers import pipeline
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm

# 1. Test-Set laden
df_test = pd.read_excel('sample_labeled.xlsx')  # oder .csv

# 2. Pipeline initialisieren
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# 3. Vorhersagen sammeln
tqdm.pandas(desc="Predicting")
results = df_test['comment_body'].progress_apply(
    lambda txt: sentiment_pipe(txt, truncation=True)[0]
)

# 4. Neues Labeling mit breiterem Neutral-Fenster
def better_label(r, pos_thr=0.7, neg_thr=0.7):
    lbl, score = r['label'].lower(), r['score']
    if lbl == 'positive' and score >= pos_thr:
        return 'positive'
    if lbl == 'negative' and score >= neg_thr:
        return 'negative'
    return 'neutral'

df_test['pred_label'] = results.apply(lambda r: better_label(r, pos_thr=0.75, neg_thr=0.75))

# 5. Metriken berechnen
y_true = df_test['sentiment']
y_pred = df_test['pred_label']

print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred, labels=['negative','neutral','positive']))
