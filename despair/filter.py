import numpy as np
import scipy.signal as signal

import despair.util as util

"""
The filter is the non-ring filter from:

Linkoping Studies in Science and Technology. Dissertations No. 379
Focus of Attention and Gaze Control for Robot Vision.

The filtes respond to line and edge events like the following:

Magnitude is significant greater than zero.

Phase is zero for light line, is pi (or -pi) for dark line, pi/2 for
dark to light edge and -pi/2 for light to dark edge.
"""

__eps = np.finfo(np.float64).eps


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

    return util.cos2(x_pi / r2) * np.exp(-1j * (x_pi / r + np.sin(x_pi / r)))


def convolve(data: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """
    Perform convolution between the complex filter coefficients 
    and the data.

    Parameters:
    data: Real data.
        filter: Complex filter.        

    Returns:
        The complex filter response, normalized by the max magnitude.
    """
    assert coeff.dtype == np.complex128
    assert data.dtype == np.float64
    assert len(data) >= len(coeff)

    response = signal.convolve(data, coeff, mode='same')

    # The signal is padded with zeros at the edges, which can produce a sharp edge.
    # Just zero a few edge pixels in the response to only have valid responses.
    radius = (len(coeff) - 1) // 2
    response[:radius] = 0.0
    response[-radius:] = 0.0

    max_magnitude = np.max(np.abs(response))

    if max_magnitude > __eps:
        return response / max_magnitude
    else:
        return response
