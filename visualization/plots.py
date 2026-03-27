import matplotlib.pyplot as plt
import pandas as pd

def plot_results(results, title="Portfolio Value Over Time"):
    """
    Plots portfolio value over time for multiple models.

    Parameters:
        results (dict): {model_name: portfolio_history_list}
        title (str): Title of the plot
    """
    plt.figure(figsize=(12, 6))

    for model_name, history in results.items():
        history = pd.Series(history).ffill()  # forward fill any NaNs
        plt.plot(history, label=model_name)

    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value ($)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()