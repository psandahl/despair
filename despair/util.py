import math
import numpy as np


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


def cos2(x: float) -> float:
    """
    Compute cos2.

    Parameters:
        x: Value to cos.

    Returns:
        The cos2 value.
    """
    return np.cos(x) ** 2.0


def max_levels(shape: tuple[int, int]) -> int:
    """
    Compute the maximum pyramid levels possible to create before
    threshold is reached.

    Parameters:
        shape: Tuple (rows, cols).

    Returns:
        The max number of levels.
    """
    _, cols = shape

    threshold = math.log2(16)
    levels = math.floor(math.log2(cols))
    if levels > threshold:
        return levels - threshold
    else:
        return 0
