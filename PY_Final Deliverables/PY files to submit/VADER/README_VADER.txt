README for Vader Codes

---

## 1. Overview
The **VADER** folder contains all scripts and notebooks that utilize the VADER sentiment analyzer. Its purpose is to evaluate manually labeled data with VADER, compare performance metrics, and then apply the model to full movie-comment datasets.

### 1.1 Structure
```
VADER/
├── Pirates1_vader.ipynb
├── Pirates2_vader.ipynb
├── Pirates3_vader.ipynb
├── Pirates4_vader.ipynb
├── Pirates5_vader.ipynb
├── twilight1_vader.ipynb
├── twilight_newmoon.ipynb
├── twilight_eclipse.ipynb
├── twilight_bd1.ipynb
├── twilight_bd2.ipynb
├── Combining_Vader.ipynb
├── all_movies_comments_vader.ipynb
└── testVADER.py
```

---

## 2. Requirements
- **Python 3.7+**
- **pip** (or another package manager)
- Required libraries (e.g., install via `pip install …`):
  - pandas
  - numpy
  - tqdm
  - nltk (if tokenization is used)
  - vaderSentiment
  - scikit-learn
  - matplotlib (if visualizations appear in the notebooks)

> **Note:** Many notebooks include Google Colab–specific imports (e.g., `from google.colab import drive`). Those lines must be removed or adapted for local execution.

---

## 3. File Descriptions and Usage

### 3.1 `testVADER.py`
**Purpose:**
1. Loads a manually labeled sample file (`sample_labeled.xlsx`).
2. Applies VADER to assign sentiment classes (negative, neutral, positive).
3. Compares VADER’s predictions against the true labels (e.g., confusion matrix).
4. Applies VADER to a large comment dataset (`Pirates 2 .csv`) and saves the results as `Pirates_with_vader_sentiment.csv`.

**Important Details & Inconsistencies:**
- Paths in the original script are Windows-specific (using backslashes). You will need to adjust them to your local environment. For example:
  ```python
  # Original line in testVADER.py
  df_full = pd.read_csv('Pirates 2 .csv')

  # For a local Linux/macOS setup:
  df_full = pd.read_csv('path/to/Pirates_2.csv')
  ```
- The filename `Pirates 2 .csv` has an extra space before `.csv`. This may cause file-not-found errors. Consider renaming it to `Pirates_2.csv` or matching the exact name when reading it.
- The script expects `sample_labeled.xlsx` to be in the working directory or at the path you specify.

**How to Run:**
```bash
python testVADER.py
```

---

### 3.2 `twilight1_vader.ipynb`
**Purpose:**
- Applies VADER to a “Twilight” comments dataset.
- Workflow: Read data → Preprocess (cleaning) → Run VADER analysis → Evaluate (accuracy, classification report, confusion matrix).
- The same code is applied to the movies: "Pirates1", "Pirates2", "Pirates3", "Pirates4", "Pirates5", "twilight_newmoon", "twilight_eclipse", "twilight_bd1", "twilight_bd2".

**Notes & Inconsistencies:**
- Some code cells import libraries (`pandas`, `nltk`, `vaderSentiment`) while others rely on Google Colab mounts. If you run this locally, remove or comment out all Colab-specific code (e.g., `drive.mount('/content/drive')`).
- Ensure the “Twilight” comments CSV (e.g., `twilight_comments.csv`) is present in your working directory.
- Check that all paths are correct and adjust them if necessary.

---

### 3.3 `all_movies_comments_vader.ipynb`
**Purpose:**
- Processes multiple movie-comment CSV files with VADER.
- Contains loops (and some hard-coded filenames) to add VADER labels to each CSV and save the outputs.

**Notes:**
- Several sections use hard-coded filenames because the original “loop” over files did not work reliably (e.g., `# Hard-code file names as loop did not work`). You can modify these sections to detect files dynamically, but be careful to preserve the correct filenames.
- Verify that each input CSV actually exists and that column names are consistent (e.g., a `comment_body` column).
- Remove any `from google.colab import drive` and `drive.mount(...)` lines if running locally.

---

### 3.4 `Combining_Vader.ipynb`
**Purpose:**
- Merges VADER results from multiple movies into a single dataset and (optionally) generates summary statistics or plots.

**Notes:**
- Starts with Colab-specific code (`from google.colab import drive`). Remove these lines for local use.
- Reads several VADER-labeled CSV files and concatenates them into one DataFrame.
- Typical column structure: `movie_id`, `comment_body`, `vader_score`, `sentiment`, etc.
- Make sure all expected CSVs are present; otherwise, the notebook will crash on missing files.