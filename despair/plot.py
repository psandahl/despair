import matplotlib.pyplot as plt
import logging
import numpy as np
import pathlib
import scipy.ndimage as ndimage

import despair.disparity as disparity
import despair.filter as filter
import despair.image as image

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


def disparity_feature_image(r: float, scale: float) -> None:
    """
    Plot disparity values for the feature image.
    """
    logger.debug(f'disparity_feature_image: radius={r} scale={scale}')

    # Get the filter coefficients for the given radius.
    coeff = filter.coeff(r)

    # Generate the feature image, and from that extract the feature signal/reference image.
    img = __feature_image(blur=True)
    reference_img = img[0, :]

    x = np.arange(len(reference_img), dtype=np.float64)

    fig = plt.figure(figsize=(8, 7))

    # Visualize the reference image as a signal.
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.grid()
    ax1.plot(x, reference_img, color='#000000')
    ax1.set_title('Reference signal')
    ax1.set_xlim(left=0.0, right=len(reference_img) - 1)

    shift_img = __global_shift_image(img.shape, scale)
    query_img = image.horizontal_shift(img, shift_img)[0, :]

    ax_qry = fig.add_subplot(4, 1, 2)
    ax_qry.grid()
    ax_qry.plot(x, query_img, color='#000000')
    ax_qry.set_title(f'Query signal disparity={scale}')
    ax_qry.set_xlim(left=0.0, right=len(query_img) - 1)

    disparity_img = np.zeros_like(reference_img)
    confidence_img = np.zeros_like(reference_img)

    disparity.line_pair(coeff, reference_img, query_img,
                        disparity_img, confidence_img)

    # Hack to filter disparity using confidence. Just for plotting.
    disparity_img = np.where(confidence_img > 0.1, disparity_img, 0.0)

    ax_disp = fig.add_subplot(4, 1, 3)
    ax_disp.grid()
    ax_disp.plot(x, disparity_img, color='#0000ff')
    ax_disp.set_title('Computed disparity (filtered)')
    ax_disp.set_xlim(left=0.0, right=len(disparity_img) - 1)

    ax_conf = fig.add_subplot(4, 1, 4)
    ax_conf.grid()
    ax_conf.plot(x, confidence_img, color='#00ff00')
    ax_conf.set_title('Computed confidence')
    ax_conf.set_xlim(left=0.0, right=len(confidence_img) - 1)

    fig.suptitle(f'Disparity plots for radius={r} shift scale={scale}')
    fig.tight_layout()
    plt.show()


def disparity_ground_truth(reference: pathlib.Path, mode: str, scale: float,
                           r: float, max_level: int) -> bool:
    logger.debug(
        f'disparity ground truth: reference={reference}, mode={mode}, scale={scale} radius={r} max_level={max_level}')

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

    query_img = image.horizontal_shift(ref_img, shift_img)

    # Hack.
    items = disparity.compute(ref_img, query_img, r, max_level)

    item = items[1]
    conf_img = item['confidence']
    disp_img = item['disparity']

    fig = plt.figure(figsize=(8, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.grid()
    ax1.imshow(disp_img, cmap='gray', vmin=np.min(
        disp_img), vmax=np.max(disp_img))
    ax1.set_title('Disparity')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.grid()
    ax2.imshow(conf_img, cmap='gray')
    ax2.set_title('Confidence')

    fig.tight_layout()
    plt.show()

    return True


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
