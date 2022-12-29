def mix(x0: float, x1: float, amount: float) -> float:
    """
    Interpolate between two values.

    Parameters:
        x0: The first value.
        x1: The second value.
        amount: The mix factor 0 - 1, tells how much to take from x1.

    Returns:
        The interpolated value.
    """
    return x0 * (1.0 - amount) + x1 * amount
