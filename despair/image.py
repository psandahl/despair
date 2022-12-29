import despair.util as util

import logging
import math
import numpy as np
import pathlib
import skimage.io as io

logger = logging.getLogger(__name__)


def read_grayscale(path: pathlib.Path) -> np.ndarray:
    """
    Read an image, converted to 64-bit floating point grayscale.

    Parameters:
        path: The file path to the image.

    Returns:
        The grayscale image.
    """
    try:
        return io.imread(path, as_gray=True)
    except FileNotFoundError as e:
        logger.error(e)
        return None


def black_grayscale(shape: tuple[int, int]) -> np.ndarray:
    """
    Create a "black" 64-bit floating point grayscale image.

    Parameters:
        shape: tuple (rows, cols).

    Returns:
        The grayscale image.
    """
    return np.zeros(shape, dtype=np.float64)


def horizontal_shift(src: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """
    Create a new image, with same dimension as src, which is shifted
    using the values in shift. Linear interpolation is used for subpixel
    shifts.

    Parameters:
        src: Source image. Must be 64-bit floating point image.
        shift: Shift image. Must be 64-bit floating point image.

    Returns:
        The shifted copy of src.
    """
    assert src.ndim == 2
    assert shift.ndim == 2
    assert len(src.shape) == 2
    assert src.shape == shift.shape
    src.dtype = np.float64
    shift.dtype = np.float64

    dest = black_grayscale(src.shape)
    rows, cols = src.shape
    y = 0
    while y < rows:
        src_row = src[y, :]
        shift_row = shift[y, :]
        dest_row = dest[y, :]

        x = 0
        while x < cols:
            shift_val = x - shift_row[x]
            x0 = math.floor(shift_val)
            x1 = math.ceil(shift_val)
            frac = shift_val - x0

            if x0 >= 0.0 and x1 < cols:
                dest_row[x] = util.mix(src_row[x0], src_row[x1], frac)

            x += 1

        y += 1

    return dest
