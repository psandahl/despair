import cmath
import numpy as np

"""
Filters are taken from:

Linkoping Studies in Science and Technology. Dissertations No. 379
Focus of Attention and Gaze Control for Robot Vision

Carl-Johan Westelius
"""


def cos2(x: float) -> float:
    """
    Compute cos2.

    Parameters:
        x: Value to cos.

    Returns:
        The cos2 value.
    """
    return np.cos(x) ** 2.0


def nonring(r: float) -> np.ndarray:
    """
    Compute a discrete, complex nonring filter with the given radius.

    Parameters:
        r: Radius of the filter.

    Returns:
        The filter coefficients, in a complex numpy array.
    """
    r2 = r * 2.0

    x = np.arange(-r, r + 1, dtype=np.float64)
    x_pi = x * np.pi

    return cos2(x_pi / r2) * np.exp(-1j * (x_pi / r + np.sin(x_pi / r)))


def wft(r: float) -> np.ndarray:
    """
    Compute a discrete, complex windowed fourier transform 
    filter with the given radius.

    Parameters:
        r: Radius of the filter.

    Returns:
        The filter coefficients, in a complex numpy array.
    """
    x = np.arange(-r, r + 1, dtype=np.float64)

    return np.exp(-1j * np.pi * (x / r))
