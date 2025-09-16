# 🎬 Sentiment Analysis of Popular Movie Sequels

This repository contains the code, data, and analysis for the project **"Sentiment Analysis of Popular Sequels based on Reddit Comments"**, developed by **Group A** (Ewald Seelich, Maelys Lafaurie, Phuong Anh, Sora Schultz, Lillian Hsiao).
This repository showcases the final group project of the course "Data Analysis and Machine Learning with Python", a course taught at the National Taiwan University (NTU).
Our goal was to analyze audience sentiment towards two major movie franchises—**Pirates of the Caribbean** and **Twilight**—and study how sentiment evolves across sequels. We compared a **lexicon-based baseline model (VADER)** with a **deep learning model (DistilBERT)**, supplemented with classical ML approaches, and explored correlations between sentiment, critics’ ratings, and box office performance.

---

## 📖 Project Overview

- **Problem**: Sequel movies often perform worse over time. We investigated whether sentiment trends on Reddit reflect this decline.  
- **Data Source**: Reddit comments collected via the Reddit API (PRAW).  
- **Models Used**:  
  - **Baseline**: VADER (lexicon-based sentiment analysis)  
  - **Custom Models**: BERT + Linear SVC, Logistic Regression  
  - **Deep Learning**: DistilBERT fine-tuning for sentiment classification  
  - **Ensemble**: Combined BERT + classical ML models with optimized weights  
- **Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrices  
- **Additional Analysis**: Correlations with Rotten Tomatoes ratings, box office revenue, and production budget.  

![Project Methodology](Archive/Group%20Presentation_Methodology.png)

![Project Demo](Archive/Group%20Presentation_Demo.png)

---

## 🛠️ Results

- **Our Ensemble Model Accuracy: ~0.546**
- **VADER Baseline Accuracy: ~0.507**
- **Key Insights:**
  - VADER consistently predicts more positive sentiment, but is less context-aware.
  - BERT-derived sentiment aligns better with critic and audience scores.
  - Pirates of the Caribbean generally scores higher sentiment than Twilight.
  - High production budgets do not necessarily correlate with higher sentiment.
 
---

 ## 🔮 Future Work

- Use larger transformer models (RoBERTa, DeBERTa).
- Expand dataset beyond Reddit (YouTube, Twitter, fan forums).
- Implement multi-label sentiment classification.
- Add thematic tagging (e.g., plot, characters, production).
- Incorporate event-based metadata (trailers, casting news, release timing).

---

## 👥 Authors

- Ewald Seelich – Baseline modeling, discussion
- Maelys Lafaurie – Data collection, introduction
- Phuong Anh – Results compilation, visualization
- Sora Schultz – Preprocessing, modeling, visualizations
- Lillian Hsiao – Data cleaning, related works, dataset integration

---

## 🙏 Acknowledgements

- Professor Cheng-Yuan Ho for guidance
- Teaching Assistants for continuous support
- The Reddit community for open data access
- Developers of open-source libraries: PRAW, pandas, scikit-learn, PyTorch, Hugging Face Transformers
