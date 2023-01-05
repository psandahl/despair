import numpy as np
import scipy.ndimage as ndimage

import despair.image as image


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
    img = image.black_grayscale((160, 160))

    img[:, 19:22] = 1.0
    img[:, 60:141] = 1.0
    img[:, 99:102] = 0.0

    if blur:
        return ndimage.gaussian_filter(img, 1.0)
    else:
        return img


def global_shift_image(shape: tuple[int, int], scale: float) -> np.ndarray:
    """
    Create a global shift image.

    Parameters:
        shape: Shape of the image.
        scale: Shift scale.

    Returns:
        The shift image.
    """
    img = image.black_grayscale(shape)
    img[:, :] = scale

    return img


def peak_shift_image(shape: tuple[int, int], scale: float) -> np.ndarray:
    """
    Create shift image with a peak in the middle.

    Parameters:
        shape: Shape of the image.
        scale: Shift scale.

    Returns:
        The shift image.
    """
    rows, cols = shape
    min_radius = (min(rows, cols) - 1) / 2
    center_x, center_y = (cols - 1) / 2, (rows - 1) / 2

    ys, xs = np.ogrid[:rows, :cols]
    img = np.cos(((ys - center_y) ** 2 + (xs - center_x) ** 2) /
                 min_radius ** 2)
    img = np.where(img > 0, img, 0.0)

    return img * scale
