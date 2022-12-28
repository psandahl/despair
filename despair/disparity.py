import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute(reference: np.ndarray, query: np.ndarray, radius: int = 7, levels: int = 0) -> list:
    """
    Compute the disparity from the reference image to the query image. The images are
    assumed to be stereo rectified.

    Parameters:
        reference: The reference image (one channel, floating point 
        image in range [0. - 1.0]).
        query: The query image. Same dimension, and same general
        assumptions as the reference image.
        radius: Radius (in pixels) for the phase filter.
        levels: The number of (extra) pyramid levels the disparity is 
        computed from.

    Returns:
        List of tuples (one tuple per pyramid/scale level). Top level is level zero.
        Each tuple is (disparity at scale, certainty at scale, reference at scale, query at scale).
        All are images of the same dimensions per scale.
    """
    assert isinstance(reference, np.ndarray)
    assert isinstance(query, np.ndarray)
    assert reference.ndim == 1
    assert query.ndim == 1
    assert len(reference.shape) == 2
    assert reference.shape == query.shape
    assert reference.dtype == np.float64
    assert reference.dtype == query.dtype
    assert radius > 0
    assert levels >= 0

    return list()
