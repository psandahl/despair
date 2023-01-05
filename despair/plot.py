import matplotlib.pyplot as plt
import logging
import math
import numpy as np
import pathlib
import scipy.ndimage as ndimage

import despair.disparity as disparity
import despair.filter as filter
import despair.image as image
import despair.util as util

logger = logging.getLogger(__name__)


def coeff(r: float) -> None:
    """
    Plot the filter coefficients.

    Parameters:
        r: Radius of the filter.
    """
    assert r > 0

    logger.debug(f'coeff: radius={r}')

    coeff = filter.coeff(r)

    x = np.arange(-r, r + 1, dtype=np.float64)

    fig = plt.figure(figsize=(8, 2))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(x, coeff.real, color='#0000ff', linewidth=2)
    ax1.grid()
    ax1.set_title('real/even part')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(x, coeff.imag, color='#00ff00', linewidth=2)
    ax2.grid()
    ax2.set_title('imaginary/odd part')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(x, coeff.real, color='#0000ff',
             linewidth=2)
    ax3.plot(x, coeff.imag, color='#00ff00',
             linewidth=2)
    ax3.grid()
    ax3.set_title(f'complex')

    fig.suptitle(f'Filter coefficients using radius={r}')
    fig.tight_layout()
    plt.show()


def response(r: float) -> None:
    """
    Plot the filter response using the given radius.

    Parameters:
        r: Radius of the filter.
    """
    assert r > 0

    logger.debug(f'response: radius={r}')

    # Generate the feature image, and from that extract the feature signal.
    image = __feature_image(blur=True)
    signal = image[0, :]

    x = np.arange(len(signal), dtype=np.float64)

    fig = plt.figure(figsize=(8, 7))

    # Visualize the feature image.
    ax1 = fig.add_subplot(5, 1, 1)
    ax1.imshow(image[:10, :], vmin=0.0, vmax=1.0, cmap='gray')
    ax1.set_title('Feature Image')

    # Visualize the feature signal.
    ax2 = fig.add_subplot(5, 1, 2)
    ax2.grid()
    ax2.plot(x, signal, color='#000000')
    ax2.set_title('Feature Signal')
    ax2.set_xlim(left=0.0, right=len(signal) - 1)

    coeff = filter.coeff(r)
    resp = filter.convolve(signal, coeff)

    # The raw, complex, response.
    ax3 = fig.add_subplot(5, 1, 3)
    ax3.plot(x, resp.real, color='#0000ff')
    ax3.plot(x, resp.imag, color='#00ff00')
    ax3.grid()
    ax3.set_title('complex response')
    ax3.set_xlim(left=0.0, right=len(signal) - 1)

    # The magnitude of the response.
    ax4 = fig.add_subplot(5, 1, 4)
    ax4.plot(x, np.abs(resp), color='#ff0000')
    ax4.grid()
    ax4.set_title('magnitude')
    ax4.set_xlim(left=0.0, right=len(signal) - 1)

    # The phase of the response.
    ax5 = fig.add_subplot(5, 1, 5)
    ax5.plot(x, np.angle(resp), color='#ff00ff')
    ax5.grid()
    ax5.set_title('phase angle')
    ax5.set_xlim(left=0.0, right=len(signal) - 1)

    fig.suptitle(f'Filter response using radius={r}')
    fig.tight_layout()
    plt.show()


def shift(reference: pathlib.Path, mode: str, scale: float) -> bool:
    """
    Plot a shifted query image using reference image, mode and scale.
    """
    logger.debug(f'shift: reference={reference}, mode={mode}, scale={scale}')

    ref_img = image.read_grayscale(reference)
    if ref_img is None:
        return False

    shift_img = None
    if mode == 'global':
        shift_img = __global_shift_image(ref_img.shape, scale)
    elif mode == 'peak':
        shift_img = __peak_shift_image(ref_img.shape, scale)
    else:
        logger.error(f"Unknown mode='{mode}'")
        return False

    fig = plt.figure(figsize=(8, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(ref_img, vmin=0.0, vmax=1.0, cmap='gray')
    ax1.set_title('Reference image')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(shift_img, vmin=np.min(shift_img),
               vmax=np.max(shift_img), cmap='gray', interpolation='nearest')
    ax2.set_title(f'Shift mode={mode}, scale={scale}')

    query_img = image.horizontal_shift(ref_img, shift_img)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(query_img, vmin=0.0, vmax=1.0, cmap='gray')
    ax3.set_title('Query image')

    fig.suptitle('Horizontal shift')
    fig.tight_layout()
    plt.show()

    return True


def disparity_feature_image(radius: float, scale: float) -> None:
    """
    Plot disparity values for the feature image.
    """
    logger.debug(f'disparity_feature_image: radius={radius} scale={scale}')

    # Generate the feature image, and from that extract the feature signal/reference image.
    reference_img = __feature_image(blur=True)
    shift_img = __global_shift_image(reference_img.shape, scale)
    query_img = image.horizontal_shift(reference_img, shift_img)

    reference_signal = reference_img[0, :]
    x = np.arange(len(reference_signal), dtype=np.float64)

    fig = plt.figure(figsize=(8, 6))

    # Visualize the reference signal.
    ax_ref = fig.add_subplot(5, 1, 1)
    ax_ref.grid()
    ax_ref.plot(x, reference_signal, color='#000000')
    ax_ref.set_title('Reference signal')
    ax_ref.set_xlim(left=0.0, right=len(reference_signal) - 1)

    # Visualize the query signal.
    query_signal = query_img[0, :]
    ax_qry = fig.add_subplot(5, 1, 2)
    ax_qry.grid()
    ax_qry.plot(x, query_signal, color='#000000')
    ax_qry.set_title(f'Query signal with disparity={scale}')
    ax_qry.set_xlim(left=0.0, right=len(query_signal) - 1)

    # Get the filter coefficients for the given radius.
    coeff = filter.coeff(radius)

    # Run the disparity computations.
    ref_resp = disparity.filter_response(coeff, reference_img)
    qry_resp = disparity.filter_response(coeff, query_img)

    frequency = disparity.local_frequency(ref_resp, qry_resp)
    confidence = disparity.confidence(ref_resp, qry_resp, frequency)
    phase_difference = disparity.phase_difference(ref_resp, qry_resp)
    phase_disparity = disparity.phase_disparity(
        phase_difference, frequency, confidence)

    # Visualize the local frequency.
    frequency_signal = frequency[0, :]
    ax_frq = fig.add_subplot(5, 1, 3)
    ax_frq.grid()
    ax_frq.plot(x, frequency_signal, color='#ff0000')
    ax_frq.set_title('Local frequency')
    ax_frq.set_xlim(left=0.0, right=len(frequency_signal) - 1)

    # Visualize the confidence.
    confidence_signal = confidence[0, :]
    ax_conf = fig.add_subplot(5, 1, 4)
    ax_conf.grid()
    ax_conf.plot(x, confidence_signal, color='#ffff00')
    ax_conf.set_title('Confidence')
    ax_conf.set_xlim(left=0.0, right=len(confidence_signal) - 1)

    # Visualize the disparity.
    disparity_signal = phase_disparity[0, :]
    ax_disp = fig.add_subplot(5, 1, 5)
    ax_disp.grid()
    ax_disp.plot(x, disparity_signal, color='#0000ff')
    ax_disp.set_title('Disparity')
    ax_disp.set_xlim(left=0.0, right=len(disparity_signal) - 1)

    fig.suptitle(f'Disparity plots for radius={radius} shift scale={scale}')
    fig.tight_layout()
    plt.show()


def disparity_single(reference: pathlib.Path, shift_mode: str, shift_scale: float,
                     radius: float, target_level: int) -> bool:
    logger.debug(
        f'disparity single: reference={reference}, shift_mode={shift_mode}, shift_scale={shift_scale} radius={radius} target_level={target_level}')

    # Setting stuff up.
    ref_img = image.read_grayscale(reference)
    if ref_img is None:
        return False

    max_levels = util.max_levels(ref_img.shape)
    if max_levels < target_level:
        logger.error(
            f'Target level={target_level} is greater than available max level={max_levels}')
        return False

    adapted_shift_scale = shift_scale * math.pow(2.0, target_level)
    logger.info(
        f'shift_scale={shift_scale} adapted for target level={target_level} is {adapted_shift_scale} on level zero')

    shift_img = None
    if shift_mode == 'global':
        shift_img = __global_shift_image(ref_img.shape, adapted_shift_scale)
    elif shift_mode == 'peak':
        shift_img = __peak_shift_image(ref_img.shape, adapted_shift_scale)
    else:
        logger.error(f"Unknown mode='{shift_mode}'")
        return False

    qry_img = image.horizontal_shift(ref_img, shift_img)

    ref_pyramid = image.scale_pyramid(ref_img, target_level)
    qry_pyramid = image.scale_pyramid(qry_img, target_level)

    ref_pyr_img = ref_pyramid[-1]
    qry_pyr_img = qry_pyramid[-1]

    __image_pair(ref_pyr_img, qry_pyr_img, radius)

    return True


def disparity_pair(reference: pathlib.Path, query: pathlib.Path, radius: float, target_level: int) -> bool:
    logger.debug(
        f'disparity pair: reference={reference}, query={query} radius={radius} target_level={target_level}')

    # Setting stuff up.
    ref_img = image.read_grayscale(reference)
    if ref_img is None:
        return False

    qry_img = image.read_grayscale(query)
    if qry_img is None:
        return False

    if ref_img.shape != qry_img.shape:
        logger.error('Reference and query does not have the same shape')
        return False

    max_levels = util.max_levels(ref_img.shape)
    if max_levels < target_level:
        logger.error(
            f'Target level={target_level} is greater than available max level={max_levels}')
        return False

    ref_pyramid = image.scale_pyramid(ref_img, target_level)
    qry_pyramid = image.scale_pyramid(qry_img, target_level)

    ref_pyr_img = ref_pyramid[-1]
    qry_pyr_img = qry_pyramid[-1]

    __image_pair(ref_pyr_img, qry_pyr_img, radius)

    return True


def __image_pair(ref_img: np.ndarray, qry_img: np.ndarray, radius: float) -> None:
    # Run the disparity computations.
    coeff = filter.coeff(radius)

    ref_resp = disparity.filter_response(coeff, ref_img)
    qry_resp = disparity.filter_response(coeff, qry_img)

    frequency = disparity.local_frequency(ref_resp, qry_resp)
    confidence = disparity.confidence(ref_resp, qry_resp, frequency)
    phase_difference = disparity.phase_difference(ref_resp, qry_resp)
    phase_disparity = disparity.phase_disparity(
        phase_difference, frequency, confidence)

    fig = plt.figure(figsize=(8, 6))

    # Visualize reference and query images.
    ax_ref = fig.add_subplot(3, 2, 1)
    ax_ref.imshow(ref_img, cmap='gray', vmin=0.0, vmax=1.0)
    ax_ref.grid()
    ax_ref.set_title('Reference image')

    ax_qry = fig.add_subplot(3, 2, 2)
    ax_qry.imshow(qry_img, cmap='gray', vmin=0.0, vmax=1.0)
    ax_qry.grid()
    ax_qry.set_title('Query image')

    # Visualize response magnitudes.
    ax_ref_mag = fig.add_subplot(3, 2, 3)
    ax_ref_mag.imshow(np.abs(ref_resp), cmap='gray', vmin=0.0, vmax=1.0)
    ax_ref_mag.grid()
    ax_ref_mag.set_title('Reference response magnitude')

    ax_qry_mag = fig.add_subplot(3, 2, 4)
    ax_qry_mag.imshow(np.abs(qry_resp), cmap='gray', vmin=0.0, vmax=1.0)
    ax_qry_mag.grid()
    ax_qry_mag.set_title('Query response magnitude')

    # Visualize confidence and disparity.
    ax_conf = fig.add_subplot(3, 2, 5)
    ax_conf.imshow(confidence, cmap='gray', vmin=0.0, vmax=1.0)
    ax_conf.grid()
    ax_conf.set_title('Confidence')

    ax_disp = fig.add_subplot(3, 2, 6)
    ax_disp.imshow(phase_disparity, cmap='gray', vmin=np.min(
        phase_disparity), vmax=np.max(phase_disparity))
    ax_disp.grid()
    ax_disp.set_title('Disparity')

    fig.suptitle(f'Disparity plots radius={radius}')
    fig.tight_layout()
    plt.show()


def __feature_image(blur: bool = False) -> np.ndarray:
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


def __global_shift_image(shape: tuple[int, int], scale: float) -> np.ndarray:
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


def __peak_shift_image(shape: tuple[int, int], scale: float) -> np.ndarray:
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
