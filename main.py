from data.data_loader import load_data
from features.feature_engineering import create_features, prepare_features
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

X_train, y_train = prepare_features(train)
X_test, y_test = prepare_features(test)

# Train models
models = train_models(X_train, y_train)

results = {}

# Run backtests
for name, model in models.items():
    sim = TradingSimulator(INITIAL_CASH)
    probs = model.predict_proba(X_test)[:, 1]

    for i, prob in enumerate(probs):
        price = float(test['Close'].iloc[i])

        if prob > BUY_THRESHOLD:
            signal = 1
            confidence = prob  # scale buy size by probability
        elif prob < SELL_THRESHOLD:
            signal = -1
            confidence = 1 - prob  # scale sell size inversely
        else:
            signal = 0
            confidence = 0

        sim.step(price, signal, confidence)  # <- must be inside loop

    history = sim.get_history()
    results[name] = history
    history = np.array(history).flatten()
    print(f"\n{name}")
    print("Return:", calculate_return(history))
    print("Sharpe:", sharpe_ratio(history))

# Plot cumulative portfolio value for all models
plot_results(results, title="Portfolio Value Comparison")