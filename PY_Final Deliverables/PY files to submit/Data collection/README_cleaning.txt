README for cleaning.py

---

## 1. Overview  
`cleaning.py` merges multiple Reddit comment CSVs, removes duplicates, separates “invalid” posts based on a predefined list of post IDs, and draws a random sample of 300 rows from the cleaned data for manual labeling. It is designed for preprocessing Reddit datasets—specifically for movies like “Twilight” and “Pirates”—to prepare them for sentiment annotation workflows.

### 1.1 Structure  
```
cleaning.py
```

---

## 2. Requirements  
- **Python 3.6+**  
- **pandas**  

Install via pip:  
```bash
pip install pandas
```

---

## 3. Configuration  
1. **Input File Paths** (edit at the top of `cleaning.py`):  
   - `/content/Twilight_relevant_comments.csv`  
   - `/content/Twilight_relevant_comments_2.csv`  
   - `/content/Twilight_relevant_comments_3.csv`  
   - `/content/pirates 5 (1).csv`  
   - `/content/pirate4_subreddits_comments.csv`  
   - `/content/pirates 5_cleaned (1).csv`  

   *If running locally, replace `/content/...` with your local paths.*

2. **Invalid Post IDs**  
   - The list `ids_to_extract` contains post IDs to be considered “invalid.”  
   - Rows with these IDs go into `pirate4_merged_extra.csv`; all others go into `pirate4_merged_cleaned.csv`.  
   - Update `ids_to_extract` as needed.

3. **Sample Size**  
   - Currently set to sample `n = 300` rows from `pirates 5_cleaned (1).csv`.  
   - To change: modify the `n=` parameter in `df.sample(...)` (near the bottom of the script).  

---

## 4. Workflow Overview  
1. **Merge “Twilight” Comments**  
   - Read three CSVs:  
     - `Twilight_relevant_comments.csv`  
     - `Twilight_relevant_comments_2.csv`  
     - `Twilight_relevant_comments_3.csv`  
   - Concatenate into one DataFrame, drop duplicate rows.  
   - Save as `Twilight_merged_comments.csv`.

2. **Inspect “Pirates 5” Extra File**  
   - Load `pirates 5 (1).csv` into `dp`.  
   - Print the total row count of `dp`.  
   - Group by `post_id` and print a mapping `post_id → post_title` to identify invalid posts.

3. **Clean “Pirates 4” Comments**  
   - Load `pirate4_subreddits_comments.csv` into `df`.  
   - Drop duplicates.  
   - Split into:  
     - `merged_extra`: rows where `post_id` is in `ids_to_extract`.  
     - `merged_cleaned`: rows where `post_id` is not in `ids_to_extract`.  
   - Save:  
     - `pirate4_merged_extra.csv` (invalid posts).  
     - `pirate4_merged_cleaned.csv` (valid, de-duplicated).  
   - Print the row count of `merged_cleaned`.

4. **Random Sampling for Labeling**  
   - Load `pirates 5_cleaned (1).csv` into `df`.  
   - Draw 300 random rows (`random_state=42` for reproducibility).  
   - Save this sample as `5-labelled.csv` for manual sentiment annotation.  
   - Print a confirmation of sample size.

---

## 5. Usage  
1. Ensure `pandas` is installed.  
2. Update all file paths at the top of `cleaning.py`.  
3. Adjust `ids_to_extract` (invalid post IDs) if needed.  
4. Change sample size by editing `n=300` in the sampling call.  
5. Run:
   ```bash
   python cleaning.py
   ```
6. Outputs generated alongside inputs (or in your working directory):  
   - `Twilight_merged_comments.csv`  
   - (Console) row count of `pirates 5 (1).csv`  
   - (Console) `post_id → post_title` list from `pirates 5 (1).csv`  
   - `pirate4_merged_extra.csv`  
   - `pirate4_merged_cleaned.csv`  
   - `5-labelled.csv`  

---

## 6. Notes & Tips  
- **Local Paths**: Replace `/content/...` with appropriate local file paths.  
- **Duplicate Logic**: Currently uses `drop_duplicates()` on entire rows. If you wish to de-duplicate by `comment_id` only, modify that line accordingly.  
- **Reproducibility**: `random_state=42` ensures the same 300-row sample each run. Change or remove for a different sample.  
- **Extensibility**:  
  - To merge additional “Twilight” files, add them to the `pd.concat([...])` list.  
  - You can parameterize file paths and sample size by adding `argparse` logic.  
  - Automate sentiment-label sampling for other movies by refactoring into functions or accepting command-line arguments.  

---

**End of README for cleaning.py**
