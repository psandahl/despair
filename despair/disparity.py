import cmath
import logging
import numpy as np

import despair.filter as filter
import despair.util as util

logger = logging.getLogger(__name__)

__eps = np.finfo(np.float64).eps


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
        Each tuple is (disparity at scale, confidence at scale, reference at scale, query at scale).
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


def line(coeff: np.ndarray, reference: np.ndarray, query: np.ndarray,
         disparity: np.ndarray, confidence: np.ndarray) -> None:
    """
    Compute the disparity between the reference and the query image lines.
    The disparity is described how the query image shall be shifted to meet
    the reference image.

    The confidence is between zero and one.

    Parameters:
        coeff: The filter coefficients.
        reference: The reference image line.
        query: The query image line.
        disparity: The disparity image [output].
        confidence: The confidence iamge [output].
    """
    assert isinstance(coeff, np.ndarray)
    assert len(coeff.shape) == 1
    assert coeff.dtype == np.complex128
    assert isinstance(reference, np.ndarray)
    assert len(reference.shape) == 1
    assert reference.dtype == np.float64
    assert isinstance(query, np.ndarray)
    assert query.shape == reference.shape
    assert query.dtype == reference.dtype
    assert isinstance(disparity, np.ndarray)
    assert disparity.shape == reference.shape
    assert disparity.dtype == reference.dtype
    assert isinstance(confidence, np.ndarray)
    assert confidence.shape == reference.shape
    assert confidence.dtype == reference.dtype

    # Compute filter responses for the reference and the query lines.
    resp_ref = filter.convolve(reference, coeff)
    resp_qry = filter.convolve(query, coeff)

    # Compute the phase difference between the responses.
    # Note: phase_diff is 'd' in report paper.
    phase_diff = resp_ref * np.conj(resp_qry)

    # And from the phase difference extract magnitudes and phase angles.
    magnitude = np.abs(phase_diff)
    angle = np.angle(phase_diff)

    # Compute the local frequency.
    local_frequency = __local_frequency(resp_ref, resp_qry)

    # Compute the confidence.
    confidence[:] = __confidence(
        resp_ref, resp_qry, magnitude, local_frequency)

    # Compute the disparity using the phase angles and the local frequencies.
    disparity[:] = angle / local_frequency


def __local_frequency(resp_ref: np.ndarray, resp_qry: np.ndarray) -> np.ndarray:
    local_frequency = np.zeros(resp_ref.shape[0], dtype=np.float64)

    idx = 0
    for idx in range(1, len(local_frequency) - 1):
        # The phase difference between the current position and its
        # two neighbors is an estimation of how fast the phase varies,
        # i.e. the local frequency.
        ref_min = resp_ref[idx - 1] * resp_ref[idx].conjugate()
        ref_plus = resp_ref[idx] * resp_ref[idx + 1].conjugate()
        qry_min = resp_qry[idx - 1] * resp_qry[idx].conjugate()
        qry_plus = resp_qry[idx] * resp_qry[idx + 1].conjugate()

        local_frequency[idx] = cmath.phase(
            ref_min + ref_plus + qry_min + qry_plus)

        idx += 1

    return np.where(np.abs(local_frequency) < __eps, __eps, local_frequency)


def __confidence(resp_ref: np.ndarray, resp_qry: np.ndarray,
                 magnitude: np.ndarray, local_frequency: np.ndarray) -> np.ndarray:
    m2 = np.abs(resp_ref * resp_qry)
    alpha = np.abs(resp_ref) * np.abs(resp_qry)
    gamma = 4.0  # Heuristic value.

    c1 = np.sqrt(m2) * np.power((2 * alpha) / (1.0 + alpha ** 2), gamma)
    c2 = c1 * util.cos2(magnitude / 2.0)
    c3 = np.where(local_frequency > 0.0, c2, 0.0)

    return c3
