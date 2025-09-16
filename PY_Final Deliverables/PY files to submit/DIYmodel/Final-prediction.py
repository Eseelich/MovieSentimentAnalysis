import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentiment_utils import (
    load_labeled_data, split_and_oversample,
    train_transformer, tune_negative_threshold,
    train_linear_svc, train_logistic_regression,
    ensemble_predict, clean_text, id2label
)
import joblib
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--movies', nargs='+', required=True)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--w_tf', type=float, default=0.5)
    parser.add_argument('--w_svc', type=float, default=0.25)
    parser.add_argument('--w_lr', type=float, default=0.25)
    args = parser.parse_args()

    selected = [m for m in args.movies if m.isdigit()]
    # Map movie IDs to file paths in 'labeled' and 'all' directories
    labeled_map = {m: os.path.join('labeled', f'{m}_labeled.xlsx') for m in selected}
    all_map     = {m: os.path.join('all',   f'{m}_all.csv')       for m in selected}

    # Load labeled
    df = load_labeled_data(labeled_map)

    # Split
    train_df, test_df = split_and_oversample(df)

    # Train models
    tr, tok, tm = train_transformer(train_df, test_df)
    neg_thr = tune_negative_threshold(tr)
    svc_vec, svc_clf = train_linear_svc(train_df, test_df)
    lr_vec, lr_clf   = train_logistic_regression(train_df, test_df)

    # Grid search ensemble weights
    best_acc, best_weights = 0, None
    for w_tf in np.linspace(0.5, 1.0, 6):
        for w_svc in np.linspace(0.0, 1.0 - w_tf, 6):
            w_lr = 1.0 - w_tf - w_svc
            preds = ensemble_predict(
                [tm, svc_clf, lr_clf],
                [tok, svc_vec, lr_vec],
                test_df['cleaned_body'].tolist(),
                neg_thr, [w_tf, w_svc, w_lr]
            )
            acc = accuracy_score(test_df['label'], preds)
            if acc > best_acc:
                best_acc, best_weights = acc, [w_tf, w_svc, w_lr]
    print(f'Best weights: DistilBERT={best_weights[0]:.2f}, SVC={best_weights[1]:.2f}, LR={best_weights[2]:.2f}')

    # Final evaluation
    preds_final = ensemble_predict(
        [tm, svc_clf, lr_clf],
        [tok, svc_vec, lr_vec],
        test_df['cleaned_body'].tolist(),
        neg_thr, best_weights
    )
    print('=== Final Ensemble Evaluation ===')
    print('Accuracy:', accuracy_score(test_df['label'], preds_final))
    print(classification_report(test_df['label'], preds_final, target_names=['negative','neutral','positive']))
    cm = confusion_matrix(test_df['label'], preds_final, labels=[0,1,2])
    disp = ConfusionMatrixDisplay(cm, display_labels=['neg','neu','pos'])
    disp.plot(cmap='Blues')
    plt.show()

    # Save artifacts
    tm.save_pretrained('distilbert_finetuned')
    tok.save_pretrained('tokenizer')
    joblib.dump(svc_clf, 'svc_clf.pkl')
    joblib.dump(lr_clf, 'lr_clf.pkl')
    with open('pipeline_results.pkl','wb') as f:
        pickle.dump({'neg_thr':neg_thr,'best_weights':best_weights,'preds':preds_final,'true':test_df['label'].tolist()}, f)

    # Label full data
    for m, path in all_map.items():
        df_all = pd.read_csv(path)
        df_all = df_all.dropna(subset=['comment_body'])
        df_all['cleaned_body'] = df_all['comment_body'].apply(clean_text)
        preds_all = ensemble_predict(
            [tm, svc_clf, lr_clf],
            [tok, svc_vec, lr_vec],
            df_all['cleaned_body'].tolist(),
            neg_thr, best_weights
        )
        df_all['predicted_sentiment'] = [id2label[p] for p in preds_all]
        df_all.to_csv(os.path.join('all', f'{m}_all_labeled.csv'), index=False)
        print(f"Labeled full dataset for movie {m} saved.")
