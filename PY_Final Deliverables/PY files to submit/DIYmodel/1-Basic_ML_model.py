# Komplettes Skript: Manuelles Under‑ & Oversampling + LinearSVC + CalibratedClassifier

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score

# 1. Gelabeltes Sample einlesen
sample = pd.read_excel(r'sample_labeled.xlsx')
X = sample['comment_body']
y = sample['sentiment']

# 2. Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Manuelles Under- & Oversampling:
train_df = pd.DataFrame({'comment_body': X_train, 'sentiment': y_train})
counts = train_df['sentiment'].value_counts()
# Ziel: alle Klassen auf Median-Anzahl bringen
target = int(counts.median())

resampled = []
for cls, grp in train_df.groupby('sentiment'):
    if len(grp) < target:
        # Oversample
        resampled.append(grp.sample(n=target, replace=True, random_state=42))
    else:
        # Undersample
        resampled.append(grp.sample(n=target, replace=False, random_state=42))
train_bal = pd.concat(resampled).sample(frac=1, random_state=42)
X_train_bal = train_bal['comment_body']
y_train_bal = train_bal['sentiment']

# 4. TF-IDF + LinearSVC (calibriert)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,3), stop_words='english')
svc = LinearSVC(class_weight='balanced', max_iter=10000)
clf = CalibratedClassifierCV(svc, cv=5)  # für probabilistische Thresholds

# 5. Pipeline-Fit
X_train_tfidf = tfidf.fit_transform(X_train_bal)
clf.fit(X_train_tfidf, y_train_bal)

# 6. Validierung
X_val_tfidf = tfidf.transform(X_val)
y_val_pred = clf.predict(X_val_tfidf)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))

# 7. Anwenden auf den gesamten Datensatz
df_full = pd.read_csv(r'Pirates 2 .csv')
X_full_tfidf = tfidf.transform(df_full['comment_body'])
df_full['sentiment'] = clf.predict(X_full_tfidf)
df_full.to_csv(r'Pirates_with_sentiment_final.csv', index=False)
print("\nFertig: 'Pirates_with_sentiment_final.csv' erstellt.")
