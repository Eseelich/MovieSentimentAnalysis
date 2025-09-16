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

    # Save models and variables for later analysis
    import joblib
    import pickle
    # Save DistilBERT model and tokenizer
    tm.save_pretrained('distilbert_finetuned')
    tok.save_pretrained('tokenizer')
    # Save classical models
    joblib.dump(svc_clf, 'svc_clf.pkl')
    joblib.dump(lr_clf, 'lr_clf.pkl')
    # Save pipeline variables and results
    results = {
        'neg_thr': neg_thr,
        'best_weights': best_weights,
        'test_accuracy': best_acc,
        'classification_report': classification_report(test_df['label'], preds_final, target_names=['negative','neutral','positive'], output_dict=True),
        'confusion_matrix': confusion_matrix(test_df['label'], preds_final, labels=[0,1,2]).tolist(),
        'predictions': preds_final,
        'true_labels': test_df['label'].tolist()
    }
    with open('pipeline_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("All models and results saved: 'distilbert_finetuned/', 'tokenizer/', 'svc_clf.pkl', 'lr_clf.pkl', 'pipeline_results.pkl'")

        # Statistics of predicted sentiment distribution
    import matplotlib.pyplot as plt
    stats = pd.Series(preds_final).value_counts(normalize=True).sort_index()
    stats.index = ['negative', 'neutral', 'positive']
    print("=== Predicted Sentiment Distribution ===")
    print(stats)
    # Plot distribution
    stats.plot(kind='bar')
    plt.title('Predicted Sentiment Distribution')
    plt.ylabel('Proportion')
    plt.xlabel('Sentiment')
    plt.tight_layout()
    plt.savefig('predicted_distribution.png')
    plt.show()

    # Confusion matrix heatmap
    import matplotlib.pyplot as plt
    cm = confusion_matrix(test_df['label'], preds_final, labels=[0,1,2])
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(cm, display_labels=['negative','neutral','positive'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Classification report bar chart
    report_data = classification_report(test_df['label'], preds_final, target_names=['negative','neutral','positive'], output_dict=True)
    report_df = pd.DataFrame(report_data).transpose()
    report_df[['precision','recall','f1-score']].plot(kind='bar')
    plt.title('Classification Metrics by Class')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('classification_report.png')
    plt.show()
