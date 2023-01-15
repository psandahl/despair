import cmath
import logging
import numpy as np

from scipy import ndimage

import despair.filter as filter
import despair.image as image
import despair.util as util

logger = logging.getLogger(__name__)


def compute(reference: np.ndarray, query: np.ndarray, radius: int, refine: bool = True):
    """
    Compute the disparity for the image pair.
    """
    assert isinstance(reference, np.ndarray)
    assert len(reference.shape) == 2
    assert reference.dtype == np.float64
    assert isinstance(query, np.ndarray)
    assert len(query.shape) == 2
    assert query.dtype == np.float64
    assert reference.shape == query.shape

    # Prepare stuff before main algorithm loop.
    coeff = filter.coeff(radius)
    pyr_levels = util.max_levels(reference.shape)
    ref_pyramid = image.scale_pyramid(reference, pyr_levels)
    qry_pyramid = image.scale_pyramid(query, pyr_levels)

    conf_accum = None
    disp_accum = None

    logger.info(
        f'compute: input image size (w, h)={reference.shape[::-1]}, pyr levels={pyr_levels}, coarsest image size={ref_pyramid[-1].shape[::-1]}')
    logger.info(
        f'compute: filter radius={radius} disparity refine each level={refine}')


def compute_pair(ref_img: np.ndarray, qry_img: np.ndarray, coeff: np.ndarray) -> dict:
    assert isinstance(ref_img, np.ndarray)
    assert len(ref_img.shape) == 2
    assert ref_img.dtype == np.float64
    assert isinstance(qry_img, np.ndarray)
    assert len(qry_img.shape) == 2
    assert qry_img.dtype == np.float64
    assert ref_img.dtype == qry_img.dtype
    assert isinstance(coeff, np.ndarray)
    assert len(coeff.shape) == 1
    assert coeff.dtype == np.complex128

    ref_resp = filter_response(coeff, ref_img)
    qry_resp = filter_response(coeff, qry_img)
    freq = local_frequency(ref_resp, qry_resp)
    conf = confidence(ref_resp, qry_resp, freq)
    phase_diff = phase_difference(ref_resp, qry_resp)
    disp = phase_disparity(phase_diff, freq, conf)
    conf_splat, disp_splat = splat(conf, disp)
    shifted_qry = image.horizontal_shift(qry_img, disp_splat)

    return {
        'reference': ref_img,
        'query': qry_img,
        'shifted_query': shifted_qry,
        'reference_response': ref_resp,
        'query_response': qry_resp,
        'frequency': freq,
        'confidence': conf,
        'disparity': disp,
        'confidence_splat': conf_splat,
        'disparity_splat': disp_splat
    }


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
            disparity[y, x] = disp_value if abs(
                disp_value) < disparity_threshold else 0.0

    return disparity


def splat(confidence: np.ndarray, disparity: np.ndarray):
    """
    Compute spatial consistency for confidence and disparity, by spreading values
    onto weak regions without.

    Parameters:
        confidence: The confidence image.
        disparity: The disparity image.

    Returns:
        Tuple (updated confidence, updated disparity).
    """
    assert isinstance(confidence, np.ndarray)
    assert len(confidence.shape) == 2
    assert confidence.dtype == np.float64
    assert isinstance(disparity, np.ndarray)
    assert len(disparity.shape) == 2
    assert disparity.dtype == np.float64
    assert confidence.shape == disparity.shape

    sigma = 1.0
    m = ndimage.gaussian_filter(
        confidence, sigma=sigma, mode='constant', cval=0.0)
    v = ndimage.gaussian_filter(
        confidence * disparity, sigma=sigma, mode='constant', cval=0.0
    )

    min_conf = 0.1
    for y, x in np.ndindex(v.shape):
        v[y, x] = v[y, x] / max(min_conf, m[y, x])

    return m, v
