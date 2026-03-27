from data.data_loader import load_data
from features.feature_engineering import create_features
from models.train_models import train_models
from backtesting.simulator import TradingSimulator
from evaluation.metrics import calculate_return, sharpe_ratio
from visualization.plots import plot_results
from config import *
import numpy as np
import pandas as pd

# Load + feature engineering
data = load_data()
data = create_features(data)

# Split
train = data[data['Date'] < TRAIN_SPLIT_DATE]
test = data[data['Date'] >= TRAIN_SPLIT_DATE]

X_train = train.drop(columns=["target", "Date", "Ticker"])
y_train = train["target"]

X_test = test.drop(columns=["target", "Date", "Ticker"])
y_test = test["target"]

# Train models
models = train_models(X_train, y_train)

results = {}

# Run backtests
for name, model in models.items():
    sim = TradingSimulator(INITIAL_CASH)
    probs = model.predict_proba(X_test)[:, 1]

    for i, prob in enumerate(probs):
        prob = float(prob)
        price = float(test['Close'].iloc[i])

        # Adjust thresholds slightly for more action
        if prob > 0.55:
            signal = 1
        elif prob < 0.45:
            signal = -1
        else:
            signal = 0

        sim.step(price, signal)

    history = sim.get_history()
    results[name] = history
    history = np.array(history).flatten()
    print(f"\n{name}")
    print("Return:", calculate_return(history))
    print("Sharpe:", sharpe_ratio(history))

# Plot cumulative portfolio value for all models
plot_results(results, title="Portfolio Value Comparison")