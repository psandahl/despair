import matplotlib.pyplot as plt
import logging
import numpy as np
import pathlib
import scipy.ndimage as ndimage

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

    # Generate the feature image, and from that extract the feature signal.
    image = __feature_image(blur=True)
    signal = image[0, :]

    x = np.arange(len(signal), dtype=np.float64)

    fig = plt.figure(figsize=(8, 7))

    img = fig.add_subplot(5, 1, 1)

    # Visualize the feature image.
    img.imshow(image[:10, :], vmin=0.0, vmax=1.0, cmap='gray')
    img.set_title('Feature Image')

    # Visualize the feature signal.
    sig = fig.add_subplot(5, 1, 2)
    sig.grid()
    sig.plot(x, signal, color='#000000')
    sig.set_title('Feature Signal')
    sig.set_xlim(left=0.0, right=len(signal) - 1)

    coeff = filter.coeff(r)
    resp = filter.convolve(signal, coeff)

    # The raw, complex, response.
    ax1 = fig.add_subplot(5, 1, 3)
    ax1.plot(x, resp.real, color='#0000ff')
    ax1.plot(x, resp.imag, color='#00ff00')
    ax1.grid()
    ax1.set_title('complex response')
    ax1.set_xlim(left=0.0, right=len(signal) - 1)

    # The magnitude of the response.
    ax2 = fig.add_subplot(5, 1, 4)
    ax2.plot(x, np.abs(resp), color='#ff0000')
    ax2.grid()
    ax2.set_title('magnitude')
    ax2.set_xlim(left=0.0, right=len(signal) - 1)

    # The phase of the response.
    ax3 = fig.add_subplot(5, 1, 5)
    ax3.plot(x, np.angle(resp), color='#ff00ff')
    ax3.grid()
    ax3.set_title('phase angle')
    ax3.set_xlim(left=0.0, right=len(signal) - 1)

    fig.suptitle(f'Filter response using radius={r}')
    fig.tight_layout()
    plt.show()


def shift(reference: pathlib.Path, mode: str, scale: float) -> bool:
    """
    Plot a shifted query image using reference image, mode and scale.
    """
    logger.debug(f'shift: reference={reference}, mode={mode}, scale={scale}')

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
