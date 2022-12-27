import numpy as np
import scipy.signal as signal

"""
The filter is the non-ring filter from:

Linkoping Studies in Science and Technology. Dissertations No. 379
Focus of Attention and Gaze Control for Robot Vision.

The filtes respond to line and edge events like the following:

Magnitude is significant greater than zero.

Phase is zero for light line, is pi (or -pi) for dark line, pi/2 for
dark to light edge and -pi/2 for light to dark edge.
"""


def coeff(r: float) -> np.ndarray:
    """
    Compute a discrete, complex non-ring filter with the given radius.

    Parameters:
        r: Radius of the filter.

    Returns:
        The filter coefficients, in a complex numpy array.
    """
    assert r > 0

    r2 = r * 2.0

    x = np.arange(-r, r + 1, dtype=np.float64)
    x_pi = x * np.pi

    return __cos2(x_pi / r2) * np.exp(-1j * (x_pi / r + np.sin(x_pi / r)))


def convolve(data: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """
    Perform convolution between the complex filter coefficients 
    and the data.

    Parameters:
    data: Real data.
        filter: Complex filter.        

    Returns:
        The complex filter response.
    """
    assert coeff.dtype == np.complex128
    assert data.dtype == np.float64
    assert len(data) >= len(coeff)

    return signal.convolve(data, coeff, mode='same')


def __cos2(x: float) -> float:
    """
    Compute cos2.

    Parameters:
        x: Value to cos.

    Returns:
        The cos2 value.
    """
    return np.cos(x) ** 2.0
