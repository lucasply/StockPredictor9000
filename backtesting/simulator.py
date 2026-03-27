# backtesting/simulator.py
import numpy as np

class TradingSimulator:
    def __init__(self, initial_cash):
        self.cash = initial_cash
        self.position = 0
        self.history = []

    def step(self, price, signal, confidence=1.0):
        """
        price: float, current price of the asset
        signal: -1 (sell), 0 (hold), 1 (buy)
        confidence: float [0,1], scales trade size
        """
        # Buy: scale by confidence and available cash
        if signal == 1 and self.cash > 0:
            buy_amount = self.cash * confidence
            self.position += buy_amount / price
            self.cash -= buy_amount

        # Sell: scale by confidence and available position
        elif signal == -1 and self.position > 0:
            sell_amount = self.position * confidence
            self.cash += sell_amount * price
            self.position -= sell_amount

        # Portfolio value
        total_value = self.cash + self.position * price
        self.history.append(total_value)

    def get_history(self):
        return np.array(self.history)