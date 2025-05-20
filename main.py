import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sentiment_utils import (
    load_labeled_data,
    split_and_oversample,
    train_transformer,
    tune_negative_threshold,
    train_linear_svc,
    ensemble_predict,
    clean_text
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate sentiment ensemble with DistilBERT and LinearSVC"
    )
    parser.add_argument(
        "--movies", nargs='+', required=True,
        help="movie IDs e.g. 5 or 1 5 7"
    )
    args = parser.parse_args()
    selected = [m for m in args.movies if m.isdigit()]

    # Define file mappings
    labeled = {m: f"{m}_labeled.xlsx" for m in selected}
    all_files = {m: f"{m}_all.csv" for m in selected}

    # Load and split data
    df = load_labeled_data(labeled)
    train_df, test_df = split_and_oversample(df)

    # Train DistilBERT transformer model
    tr, tok, tm = train_transformer(
        train_df, test_df,
        output_dir="tx_base"
    )
    neg_thr = tune_negative_threshold(tr)

    # Evaluate transformer on test set
    print("=== Transformer Evaluation ===")
    preds_tf = tr.predict(tr.eval_dataset).predictions.argmax(axis=1)
    print("Accuracy:", np.mean(preds_tf == tr.predict(tr.eval_dataset).label_ids))
    print(classification_report(
        tr.predict(tr.eval_dataset).label_ids,
        preds_tf,
        target_names=["negative","neutral","positive"]
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(
        tr.predict(tr.eval_dataset).label_ids,
        preds_tf,
        labels=[0,1,2]
    ))

    # Train LinearSVC model
    vec, svc = train_linear_svc(train_df, test_df)

    # Evaluate LinearSVC on test set
    print("=== LinearSVC Evaluation ===")
    X_test = vec.transform(test_df['cleaned_body'])
    y_test = test_df['label']
    probs_svc = svc.predict_proba(X_test)
    preds_svc = np.argmax(probs_svc, axis=1)
    print("Accuracy:", np.mean(preds_svc == y_test))
    print(classification_report(
        y_test,
        preds_svc,
        target_names=["negative","neutral","positive"]
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(
        y_test,
        preds_svc,
        labels=[0,1,2]
    ))

    # Ensemble predictions and statistics
    print("=== Ensemble Evaluation on All Comments ===")
    for m, path in all_files.items():
        df_all = pd.read_csv(path)
        df_all['cleaned_body'] = df_all['comment_body'].apply(clean_text)
        texts = df_all['cleaned_body'].tolist()
        preds_ens = ensemble_predict(
            tm, tok, svc, vec,
            texts, neg_thr
        )
        vc = pd.Series(preds_ens).value_counts(normalize=True)
        print(f"Movie {m} sentiment distribution:")
        print(vc.rename(index={0:'negative',1:'neutral',2:'positive'}))
    
    # Optionally evaluate ensemble on test set
    print("=== Ensemble Evaluation on Test Set ===")
    texts_test = test_df['cleaned_body'].tolist()
    preds_ens_test = ensemble_predict(
        tm, tok, svc, vec,
        texts_test, neg_thr
    )
    print("Accuracy:", np.mean(preds_ens_test == test_df['label']))
    print(classification_report(
        test_df['label'],
        preds_ens_test,
        target_names=["negative","neutral","positive"]
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(
        test_df['label'],
        preds_ens_test,
        labels=[0,1,2]
    ))
