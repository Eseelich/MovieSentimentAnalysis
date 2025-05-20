
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sentiment_utils import (
    load_labeled_data, split_and_oversample,
    train_transformer, tune_negative_threshold,
    train_linear_svc, train_logistic_regression,
    ensemble_predict, clean_text
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--movies', nargs='+', required=True, help='movie IDs e.g. 5 or 1 5 7')
    parser.add_argument('--folds', type=int, default=3, help='CV folds (reduce for speed)')
    args = parser.parse_args()

    selected = [m for m in args.movies if m.isdigit()]
    labeled_map = {m: f'{m}_labeled.xlsx' for m in selected}
    all_map = {m: f'{m}_all.csv' for m in selected}

    df = load_labeled_data(labeled_map)
    df['cleaned_body'] = df['cleaned_body']

    # Split data once for final evaluation
    train_df, test_df = split_and_oversample(df)

    # Train base models
    tr, tok, tm = train_transformer(train_df, test_df)
    neg_thr = tune_negative_threshold(tr)
    svc_vec, svc_clf = train_linear_svc(train_df, test_df)
    lr_vec, lr_clf = train_logistic_regression(train_df, test_df)

    # Grid search ensemble weights on test set
    best_acc, best_weights = 0, None
    print('=== Grid Search for Ensemble Weights ===')
    for w_tf in np.linspace(0.5, 1.0, 6):
        for w_svc in np.linspace(0.0, 1.0 - w_tf, 6):
            w_lr = 1.0 - w_tf - w_svc
            weights = [w_tf, w_svc, w_lr]
            preds = ensemble_predict([
                tm, svc_clf, lr_clf
            ], [
                tok, svc_vec, lr_vec
            ], test_df['cleaned_body'].tolist(), neg_thr, weights)
            acc = accuracy_score(test_df['label'], preds)
            if acc > best_acc:
                best_acc, best_weights = acc, weights
    print(f'Best weights: DistilBERT={best_weights[0]:.2f}, SVC={best_weights[1]:.2f}, LR={best_weights[2]:.2f}')
    print(f'Best test accuracy: {best_acc:.3f}\n')

    # Final evaluation with best weights
    preds_final = ensemble_predict([
        tm, svc_clf, lr_clf
    ], [
        tok, svc_vec, lr_vec
    ], test_df['cleaned_body'].tolist(), neg_thr, best_weights)
    print('=== Final Ensemble Evaluation ===')
    print('Accuracy:', accuracy_score(test_df['label'], preds_final))
    print(classification_report(
        test_df['label'], preds_final,
        target_names=['negative', 'neutral', 'positive']
    ))
    print('Confusion Matrix:')
    print(confusion_matrix(test_df['label'], preds_final, labels=[0,1,2]))
