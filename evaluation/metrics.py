import numpy as np

def calculate_return(values):
    if len(values) < 2:
        return 0
    return (values[-1] - values[0]) / values[0]

def sharpe_ratio(values):
    values = np.array(values).flatten()
    returns = np.diff(values) / values[:-1]
    if returns.std() == 0:
        return 0
    return np.mean(returns) / np.std(returns) * np.sqrt(252)