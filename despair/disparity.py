import cmath
import logging
import numpy as np

import despair.filter as filter

logger = logging.getLogger(__name__)


def filter_response(coeff: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Compute the complex filter response of an image.

    Parameters:
        coeff: The filter coefficients.
        img: The image.

    Returns:
        The response image.
    """
    assert isinstance(coeff, np.ndarray)
    assert len(coeff.shape) == 1
    assert coeff.dtype == np.complex128
    assert isinstance(img, np.ndarray)
    assert len(img.shape) == 2
    assert img.dtype == np.float64

    response = np.zeros(img.shape, dtype=np.complex128)
    rows, _ = response.shape

    row = 0
    while row < rows:
        response[row, :] = filter.convolve(img[row, :], coeff)
        row += 1

    return response


def local_frequency(ref_resp: np.ndarray, qry_resp: np.ndarray) -> np.ndarray:
    """
    Compute the local frequency (the phase derivative) between two responses.

    Parameters:
        ref_resp: The response for the reference image.
        qry_resp: The response for the query image.

    Returns:
        The frequency image.
    """
    assert isinstance(ref_resp, np.ndarray)
    assert len(ref_resp.shape) == 2
    assert ref_resp.dtype == np.complex128
    assert isinstance(qry_resp, np.ndarray)
    assert len(qry_resp.shape) == 2
    assert qry_resp.dtype == np.complex128
    assert ref_resp.shape == qry_resp.shape

    frequency = np.zeros(ref_resp.shape, dtype=np.float64)
    _, cols = ref_resp.shape

    for y, x in np.ndindex(ref_resp.shape):
        if x > 0 and x < cols - 1:
            ref_m1 = ref_resp[y, x - 1] * ref_resp[y, x].conjugate()
            ref_p1 = ref_resp[y, x] * ref_resp[y, x + 1].conjugate()
            qry_m1 = qry_resp[y, x - 1] * qry_resp[y, x].conjugate()
            qry_p1 = qry_resp[y, x] * qry_resp[y, x + 1].conjugate()

            frequency[y, x] = cmath.phase(ref_m1 + ref_p1 + qry_m1 + qry_p1)

    return frequency


def confidence(ref_resp: np.ndarray, qry_resp: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    """
    Compute the confidence between two responses. Basic rules:
    - The minimum magnitude of the responses must exceed a threshold.
    - The frequence must be greater than zero.
    - If the above holds, the confidence is the ratio between the min and max magnitude.

    Parameters:
        ref_resp: The response for the reference image.
        qry_resp: The response for the query image.
        frequence: The frequency image.

    Returns:
        The confidence image in range [0.0 - 1.0].
    """
    assert isinstance(ref_resp, np.ndarray)
    assert len(ref_resp.shape) == 2
    assert ref_resp.dtype == np.complex128
    assert isinstance(qry_resp, np.ndarray)
    assert len(qry_resp.shape) == 2
    assert qry_resp.dtype == np.complex128
    assert isinstance(frequency, np.ndarray)
    assert len(frequency.shape) == 2
    assert frequency.dtype == np.float64
    assert ref_resp.shape == qry_resp.shape
    assert ref_resp.shape == frequency.shape

    mag_treshold = 0.2

    conf = np.zeros(ref_resp.shape, dtype=np.float64)
    for y, x in np.ndindex(ref_resp.shape):
        ref_mag = abs(ref_resp[y, x])
        qry_mag = abs(qry_resp[y, x])
        if min(ref_mag, qry_mag) > mag_treshold and frequency[y, x] > 0.0:
            conf[y, x] = min(ref_mag, qry_mag) / max(ref_mag, qry_mag)

    return conf


def phase_difference(ref_resp: np.ndarray, qry_resp: np.ndarray) -> np.ndarray:
    """
    Compute the phase difference between two responses.

    Parameters:
        ref_resp: The response for the reference image.
        qry_resp: The response for the query image.

    Returns:
        The complex phase difference.
    """
    assert isinstance(ref_resp, np.ndarray)
    assert len(ref_resp.shape) == 2
    assert ref_resp.dtype == np.complex128
    assert isinstance(qry_resp, np.ndarray)
    assert len(qry_resp.shape) == 2
    assert qry_resp.dtype == np.complex128
    assert ref_resp.shape == qry_resp.shape

    return ref_resp * np.conj(qry_resp)


def phase_disparity(phase_difference: np.ndarray, frequency: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """
    Compute the phase disparity from phase difference and local frequency. Basic rules:
    - Disparity is only computed if the confidence is above a threshold.
    - The computed disparity must be below a threshold.

    Parameters:
        phase_difference: The phase difference between two responses.
        frequency: The local frequency image.
        confidence: The confidence image.

    Returns:
        The disparity image.
    """
    assert isinstance(phase_difference, np.ndarray)
    assert len(phase_difference.shape) == 2
    assert phase_difference.dtype == np.complex128
    assert isinstance(frequency, np.ndarray)
    assert len(frequency.shape) == 2
    assert frequency.dtype == np.float64
    assert isinstance(confidence, np.ndarray)
    assert len(confidence.shape) == 2
    assert confidence.dtype == np.float64
    assert phase_difference.shape == frequency.shape
    assert phase_difference.shape == confidence.shape

    disparity_threshold = 10.0
    confidence_threshold = 0.1

    disparity = np.zeros(phase_difference.shape, dtype=np.float64)
    for y, x in np.ndindex(phase_difference.shape):
        if confidence[y, x] > confidence_threshold:
            disp_value = cmath.phase(phase_difference[y, x]) / frequency[y, x]
            disparity[y, x] = disp_value if disp_value < disparity_threshold else 0.0

    return disparity
