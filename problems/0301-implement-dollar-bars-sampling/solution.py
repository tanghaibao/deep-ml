def dollar_bars(trades: list, dollar_threshold: float) -> list:
    """
    Generate dollar bars from trade data.
    
    Args:
        trades: List of tuples (price, volume) representing individual trades
        dollar_threshold: Dollar amount threshold for creating a new bar
        
    Returns:
        List of dollar bars as tuples (open, high, low, close, volume, dollar_value)
    """
    # Your code here
    cur = 0
    cur_trades = []
    res = []
    for price, volume in trades:
        if cur >= dollar_threshold:
            prices, volumes = zip(*cur_trades)
            open = prices[0]
            high = max(prices)
            low = min(prices)
            close = prices[-1]
            vol = sum(volumes)
            res.append((open, high, low, close, vol, cur))
            cur = 0
            cur_trades = []
        cur += price * volume
        cur_trades.append((price, volume))
    if cur >= dollar_threshold:
        prices, volumes = zip(*cur_trades)
        open = p