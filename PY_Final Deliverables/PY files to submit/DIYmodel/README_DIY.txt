README for DIY Codes

---

## 1. Overview
The **DIYmodel** folder contains a custom pipeline to train, evaluate, and deploy sentiment-analysis models on movie comments. It combines classical machine-learning approaches (TF-IDF + LinearSVC/LogisticRegression) with fine-tuned transformer models (DistilBERT).

### 1.1 Structure
```
DIYmodel/
├── 1-Basic_ML_model.py
├── 2-finetune_sentiment.py
├── 3-FineTunedDistilBERT.py
├── Final-model.py
├── Final-prediction.py
├── sentiment_utils.py
└── checking_labeled_columns.py
```

---

## 2. Requirements
- **Python 3.7+**
- **pip** (or another package manager)
- Required libraries (install via `pip install …`):
  - pandas
  - numpy
  - scikit-learn
  - torch (PyTorch)
  - transformers
  - scipy
  - matplotlib
  - joblib
  - openpyxl (for reading Excel files)

> **Note:** Some scripts (especially `checking_labeled_columns.py`) were originally written for Google Colab. Remove or adapt Colab-specific lines (`drive.mount`) for local use.

---

## 3. File Descriptions and Usage

### 3.1 `sentiment_utils.py`
**Purpose:**
Provides helper functions used throughout the pipeline:
- `load_labeled_data(...)` – Load and preprocess manually labeled data.
- `clean_text(text)` – Remove unwanted characters, numbers, and stopwords.
- `split_and_oversample(...)` – Manually undersample/oversample to address class imbalance.
- `train_transformer(...)` – Fine-tune DistilBERT.
- `tune_negative_threshold(...)` – Optimize threshold-based classification to better separate negative class.
- `train_linear_svc(...)` and `train_logistic_regression(...)` – Train classical ML models on TF-IDF features.
- `ensemble_predict(...)` – Combine predictions from Transformer, SVC, and LR into an ensemble decision.

**Notes & Inconsistencies:**
- Some functions expect specific file or column names. Double-check that your actual files match the expected format.

---

### 3.2 `1-Basic_ML_model.py`
**Purpose:**
Builds a simple ML model using TF-IDF features and LinearSVC to quickly evaluate sentiment classification.

**Workflow:**
1. Read a labeled sample from an Excel file.
2. Perform train/validation split (via `train_test_split`).
3. Vectorize text with TF-IDF.
4. Use `split_and_oversample(...)` to balance classes.
5. Run `GridSearchCV` on LinearSVC with calibration (`CalibratedClassifierCV`).
6. Evaluate on the validation set (accuracy, classification report).
7. Apply the trained model to a full dataset (`Pirates 2 .csv`) and save results as `Pirates_with_sentiment_final.csv`.

**How to Run:**
```bash
python 1-Basic_ML_model.py
```

---

### 3.3 `2-finetune_sentiment.py`
**Purpose:**
Fine-tunes a DistilBERT model (e.g., `distilbert-base-uncased`) for a 3-class sentiment task.

**Workflow:**
1. Load labeled sample data (from Excel or CSV).
2. Perform train/validation split.
3. Define a custom `torch.utils.data.Dataset` class.
4. Configure training arguments (batch size, learning rate, epochs, etc.).
5. Instantiate the Hugging Face `Trainer` and train the model.
6. Evaluate on the validation set (accuracy, F1 score).
7. Use the trained pipeline to predict on the full dataset (`Pirates 2 .csv`).
8. Save results as `Pirates_with_custom_sentiment.csv`.

**How to Run:**
```bash
python 2-finetune_sentiment.py
```

---

### 3.4 `3-FineTunedDistilBERT.py`
**Purpose:**
Similar to `2-finetune_sentiment.py`, but adds threshold-based inference for the negative class (to better separate negative sentiment).

**Workflow Differences:**
- After obtaining softmax probabilities for each class, a function `apply_thresh(p)` applies predetermined thresholds (`neg_thr`, `pos_thr`) to decide final labels.
- Outputs a CSV named `Pirates_with_custom_sentiment.csv`.

**Notes & Inconsistencies:**
- The code expects threshold values (`neg_thr`, `pos_thr`) to be defined. Make sure these thresholds are set or read from a config.

**How to Run:**
```bash
python 3-FineTunedDistilBERT.py
```

---

### 3.5 `Final-model.py`
**Purpose:**
Main script to train an ensemble of models (Transformer, LinearSVC, Logistic Regression) using cross-validation.

**Command-Line Arguments:**
- `--movies`: List of movie IDs to include, e.g., `--movies 1 5 7`.
- `--folds`: Number of CV folds (default: 3).

**Workflow:**
1. Load labeled data for each specified movie using `load_labeled_data`.
2. For each CV fold:
   - Balance classes via `split_and_oversample`.
   - Train DistilBERT (`train_transformer`), returning model and tokenizer.
   - Train LinearSVC (`train_linear_svc`), returning the fitted model and TF-IDF vectorizer.
   - Train Logistic Regression (`train_logistic_regression`), returning model and vectorizer.
   - Determine optimal thresholds using `tune_negative_threshold`.
   - Use `ensemble_predict` to evaluate on the validation set.
3. Save:
   - Confusion matrix plot (`confusion_matrix.png`).
   - Bar chart of classification report (`classification_report.png`).
4. Persist trained models and vectorizers under `models/` and `vectorizers/` directories.

**Notes:**
- The script assumes there are labeled files named like `movie_<ID>_labeled.csv` or similar. Confirm that your filenames match what `load_labeled_data` expects.
- Ensure that your working directory has write permissions, as the script will attempt to create subfolders `models/transformer/…`, `models/svc/…`, `models/lr/…`, and corresponding `vectorizers/` directories.

**How to Run (Example):**
```bash
python Final-model.py --movies 1 5 7 --folds 3
```

---

### 3.6 `Final-prediction.py`
**Purpose:**
Loads the trained ensemble models (Transformer, SVC, LR) and applies them to full comment datasets for each movie.

**Key Details:**
- A dictionary `all_map` in the script maps each movie ID to its raw comment CSV path, for example:
  ```python
  all_map = {
      1: 'path/to/movie1_comments.csv',
      5: 'path/to/movie5_comments.csv',
      …
  }
  ```
- Workflow:
  1. Iterate over each movie in `all_map`.
  2. Read the raw data with `pd.read_csv(path)`, dropping empty comments.
  3. Clean the text using `clean_text(...)`.
  4. Use `ensemble_predict(...)` to generate final sentiment labels.
  5. Save a new CSV (`{movie_id}_all_labeled.csv`) under the `all/` folder with an added column `predicted_sentiment`.

**Notes:**
- You must ensure these folders already exist:
  - `models/transformer/…`
  - `models/svc/…`
  - `models/lr/…`
  - `vectorizers/svc/…`
  - `vectorizers/lr/…`
  - An output folder `all/` must be present or created.
- The mapping `id2label` must correctly translate numeric predictions (e.g., 0, 1, 2) to the label strings (“negative”, “neutral”, “positive”).

**How to Run:**
```bash
python Final-prediction.py
```
(No additional arguments are needed, since the paths are hard-coded in `all_map`.)

---

### 3.7 `checking_labeled_columns.py`
**Purpose:**
Utility script for interactive inspection and editing of labeled Excel data in Google Colab. It mounts Google Drive, reads an Excel file, displays unique values in certain columns, and saves any edits back to Drive.

**Notes:**
- This script is strictly for Colab: it contains `drive.mount('/content/drive')`. If you wish to run it locally, remove those lines and adjust file paths accordingly.
- It is not necessary for the core sentiment pipeline; it only helps with manual label-checking and column inspection.