import numpy as np

def volume_bars_sampling(prices: np.ndarray, volumes: np.ndarray, volume_threshold: float) -> list:
    """
    Generate volume bars from tick/trade data.
    
    Args:
        prices: Array of trade prices for each trade/tick
        volumes: Array of trade volumes corresponding to each price
        volume_threshold: Volume threshold that triggers formation of a new bar
    
    Returns:
        List of bars, where each bar is [open, high, low, close, total_volume]
        All values rounded to 4 decimal places
    """
    def get_bar(cur, cur_prices):
        open = cur_prices[0]
        high = max(cur_prices)
        low = min(cur_prices)
        close = cur_prices[-1]
        total_volume = cur
        return np.array([open, high, low, close, total_volume]).round(4).tolist()

    ans = []
    cur = 0
    cur_prices = []
    for price, volume in zip(prices + [0], volumes + [0]):
        cur += volume
        cur_prices.append(price)
        if cur >= volume_thr