import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def load_data(path):
    """
    Load historical movie data.
    Expects a CSV with columns:
    - series: 'Pirates' or 'Twilight'
    - mean_sentiment: float
    - budget_musd: numeric (budget in millions USD)
    - rating_pct: float (rating in percent)
    - box_office_musd: numeric (global gross in millions USD)
    """
    df = pd.read_csv(path)
    # Drop rows with missing essential data
    df = df.dropna(subset=['series', 'mean_sentiment', 'budget_musd', 'rating_pct', 'box_office_musd'])
    return df

def preprocess(df):
    """
    Encode series and split features/target.
    """
    df = df.copy()
    # Binary encode series: Pirates=0, Twilight=1
    df['series_binary'] = df['series'].map({'Pirates': 0, 'Twilight': 1})
    X = df[['series_binary', 'mean_sentiment', 'budget_musd', 'rating_pct']]
    y = df['box_office_musd']
    return X, y

def train_model(X, y):
    """Train a RandomForestRegressor and report CV scores."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f'Cross-validated R² scores: {scores}')
    print(f'Mean CV R²: {np.mean(scores):.3f}')
    model.fit(X, y)
    return model

def predict_new(model, series, mean_sentiment, budget_musd, rating_pct):
    """Predict box office gross for a new movie in millions USD."""
    series_binary = 0 if series.lower() == 'pirates' else 1
    X_new = np.array([[series_binary, mean_sentiment, budget_musd, rating_pct]])
    pred = model.predict(X_new)[0]
    return pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict box office (in millions USD) for new Pirates/Twilight movies.'
    )
    parser.add_argument('--data', required=True,
                        help='Path to CSV (historical_full.csv).')
    parser.add_argument('--new_pirates', nargs=3, type=float, metavar=('MEAN_SENT', 'BUDGET_MUSD', 'RATING_PCT'),
                        help='Features for a new Pirates movie: mean_sentiment, budget_musd, rating_pct.')
    parser.add_argument('--new_twilight', nargs=3, type=float, metavar=('MEAN_SENT', 'BUDGET_MUSD', 'RATING_PCT'),
                        help='Features for a new Twilight movie: mean_sentiment, budget_musd, rating_pct.')
    args = parser.parse_args()

    # Load and train
    df = load_data(args.data)
    X, y = preprocess(df)
    model = train_model(X, y)

    # Predictions
    preds = {}
    if args.new_pirates:
        pred_p = predict_new(model, 'Pirates', *args.new_pirates)
        print(f'Predicted box office for new Pirates movie: ${pred_p*1e6:,.2f} USD')
        preds['Pirates'] = pred_p
    if args.new_twilight:
        pred_t = predict_new(model, 'Twilight', *args.new_twilight)
        print(f'Predicted box office for new Twilight movie: ${pred_t*1e6:,.2f} USD')
        preds['Twilight'] = pred_t

    # Comparison
    if 'Pirates' in preds and 'Twilight' in preds:
        winner = 'Pirates' if preds['Pirates'] > preds['Twilight'] else 'Twilight'
        print(f'=> The new {winner} movie is predicted to be more successful.')
