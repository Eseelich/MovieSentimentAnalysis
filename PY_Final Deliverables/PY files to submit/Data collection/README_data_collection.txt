README for data_collection.py

---

## 1. Overview  
`data_collection.py` automates the retrieval of Reddit comments from specified subreddits based on a search query. It uses PRAW (Python Reddit API Wrapper) to fetch top submissions matching the query, traverses all comments under those submissions, and saves relevant metadata (subreddit, post ID, post title, comment ID, timestamp, comment body, author) into a CSV file. This facilitates downstream sentiment analysis or any text-mining pipeline on Reddit data.

### 1.1 Structure  
```
data_collection.py
```

---

## 2. Requirements  
- **Python 3.6+**  
- **praw** (Python Reddit API Wrapper)  
- **pandas**  
- Standard library modules: `datetime`, `time`, `os`  

Install via pip:  
```bash
pip install praw pandas
```

---

## 3. Configuration  
1. **Reddit API Credentials**  
   - Create an app at https://www.reddit.com/prefs/apps  
   - Obtain:  
     - `client_id`  
     - `client_secret`  
     - `user_agent`  
   - Edit the `reddit = praw.Reddit(...)` block at the top of the script and insert your credentials.

2. **Search Parameters** (at the top of the script)  
   - `subreddits`: List of subreddit names (e.g., `['movies', 'twilight', 'piratesofthecaribbean']`).  
   - `query`: String to search for in each subreddit (e.g., `"Dead Men Tell No Tales"`).  
   - `limit`: Maximum number of submissions to fetch per subreddit.

3. **Output Path**  
   - By default, results are written to `pirate5_subreddits_comments.csv` in the current working directory.  
   - To change, modify the filename/path in the `df.to_csv(...)` call at the bottom of the script.

---

## 4. Usage  
1. Ensure dependencies are installed (`praw`, `pandas`).  
2. Update Reddit credentials and search parameters in `data_collection.py`.  
3. Run:
   ```bash
   python data_collection.py
   ```
4. The script will print progress (which submission is being processed).  
5. At completion, you’ll find a CSV file (default: `pirate5_subreddits_comments.csv`) containing:  
   - `subreddit`  
   - `post_id`  
   - `post_title`  
   - `comment_id`  
   - `comment_time` (UTC, format `YYYY-MM-DD HH:MM:SS`)  
   - `comment_body`  
   - `author`  

---

## 5. How It Works (High-Level Steps)  
1. **Initialize** PRAW with your credentials.  
2. **Loop** through each entry in `subreddits`.  
3. **Search** top `limit` submissions matching `query` (sorted by relevance).  
4. For each submission:  
   - Call `submission.comments.replace_more(limit=None)` to load all nested comments.  
   - Iterate through each comment, extract metadata, and append to a list.  
5. Convert the list of comment-dictionaries into a Pandas DataFrame.  
6. Save the DataFrame as a CSV (`pirate5_subreddits_comments.csv` by default).

---

## 6. Notes & Tips  
- **Rate Limiting**: PRAW auto–respects Reddit’s API limits. If fetching large volumes, consider inserting `time.sleep(...)` between requests to avoid throttling.  
- **Local vs. Colab**: If running locally, ensure the working directory is writable or specify an absolute path for the output CSV.  
- **Error Handling**: It’s recommended to wrap Reddit calls in `try/except` to catch network issues or invalid subreddit names.  
- **Extensibility**:  
  - Convert parameters to command-line arguments (via `argparse`) for flexibility.  
  - Adjust which fields to capture (e.g., upvotes, reply count) by modifying the comment-dictionary.  
  - If only top-level comments are needed, remove nested-comment traversal.  

---

**End of README for data_collection.py**
