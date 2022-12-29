import logging
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
