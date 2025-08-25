def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

    Returns:
        float: The x-coordinate of the minimum point.
    """
    # Your code here
    # 4 * x ** 3 - 9 x ** 2 = 0
    # 4 * x - 9 = 0
    # x = 9 / 4
    def ev(x):
        return x ** 4 - 3 * x ** 3 + 2
    
    return min([(ev(x), x) for x in (0, 9/4)])[-1]