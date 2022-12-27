import numpy as np
import scipy.ndimage as ndimage


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


def feature_image(blur: bool = False) -> np.ndarray:
    """
    Create a 160x160 feature image with:
    1. White line (at 20).
    2. Black to white edge (at 60).
    3. Black line (at 100).
    4. White to black edge (at 140).

    Parameters:
        blur: Flag to request gaussian blur of image.

    Returns:
        The image.
    """
    image = np.zeros((160, 160), dtype=np.float64)

    image[:, 19:22] = 1.0
    image[:, 60:141] = 1.0
    image[:, 99:102] = 0.0

    if blur:
        return ndimage.gaussian_filter(image, 1.0)
    else:
        return image
