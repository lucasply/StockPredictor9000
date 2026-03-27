class TradingSimulator:
    def __init__(self, initial_cash):
        self.cash = initial_cash
        self.position = 0  # number of shares held
        self.history = []

    def step(self, price, signal):
        """
        signal: 1 = buy, -1 = sell, 0 = hold
        """
        if signal == 1 and self.cash > 0:
            # Buy as many shares as possible with available cash
            self.position = self.cash / price
            self.cash = 0
        elif signal == -1 and self.position > 0:
            # Sell all shares
            self.cash = self.position * price
            self.position = 0
        # Hold does nothing

        # Track total portfolio value
        total_value = self.cash + self.position * price
        self.history.append(total_value)

    def get_history(self):
        return self.history