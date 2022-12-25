import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage

import despair.filter as filter


def filters(r: float) -> None:
    """
    Plot the filters using the given radius.

    Parameters:
        r: Radius of the filter.
    """
    assert r > 0

    x = np.arange(-r, r + 1, dtype=np.float64)

    filters = [
        ("Nonring", filter.nonring),
        ("Windowed FT", filter.wft)
    ]

    num_filters = len(filters)
    fig = plt.figure(figsize=(8, num_filters * 2))

    idx = 1
    for label, func in filters:
        filt = func(r)

        ax1 = fig.add_subplot(num_filters, 3, idx)
        ax1.plot(x, filt.real, color='#0000ff',
                 linewidth=2)
        ax1.grid()
        ax1.set_title(f'{label} re/even part')

        ax2 = fig.add_subplot(num_filters, 3, idx + 1)
        ax2.plot(x, filt.imag, color='#00ff00',
                 linewidth=2)
        ax2.grid()
        ax2.set_title(f'{label} im/odd part')

        ax3 = fig.add_subplot(num_filters, 3, idx + 2)
        ax3.plot(x, filt.real, color='#0000ff',
                 linewidth=2)
        ax3.plot(x, filt.imag, color='#00ff00',
                 linewidth=2)
        ax3.grid()
        ax3.set_title(f'{label} complex')

        idx += 3

    fig.suptitle(f'Filters with radius={r}')
    fig.tight_layout()
    plt.show()


def responses(r: float) -> None:
    """
    Plot the filter responses using the given radius.

    Parameters:
        r: Radius of the filter.
    """
    assert r > 0

    filters = [
        ("Nonring", filter.nonring),
        ("Windowed FT", filter.wft)
    ]

    # Generate the feature image, and from that the feature signal.
    image = feature_image()
    signal = image[0, :]

    x = np.arange(len(signal), dtype=np.float64)

    fig = plt.figure(figsize=(8, 3 + len(filters) * 4))

    fig_rows = 2 + len(filters) * 3
    img = fig.add_subplot(fig_rows, 1, 1)

    # Visualize the feature image.
    img.imshow(image[:10, :], vmin=0.0, vmax=1.0, cmap='gray')
    img.set_title('Feature Image')

    # Visualize the feature signal.
    sig = fig.add_subplot(fig_rows, 1, 2)
    sig.grid()
    sig.plot(x, signal, color='#000000')
    sig.set_title('Feature Signal')
    sig.set_xlim(left=0.0, right=len(signal) - 1)

    # For each filter, for the given, radius display its response.
    idx = 3
    for label, func in filters:
        filt = func(r)
        resp = filter.convolve(filt, signal)

        # The raw, complex, response.
        ax1 = fig.add_subplot(fig_rows, 1, idx)
        ax1.plot(x, resp.real, color='#0000ff')
        ax1.plot(x, resp.imag, color='#00ff00')
        ax1.grid()
        ax1.set_title(f'{label}({r}): complex response')
        ax1.set_xlim(left=0.0, right=len(signal) - 1)

        # The magnitude of the response.
        ax2 = fig.add_subplot(fig_rows, 1, idx + 1)
        ax2.plot(x, np.abs(resp), color='#ff0000')
        ax2.grid()
        ax2.set_title(f'{label}({r}): magnitude')
        ax2.set_xlim(left=0.0, right=len(signal) - 1)

        # The phase of the response.
        ax3 = fig.add_subplot(fig_rows, 1, idx + 2)
        ax3.plot(x, np.angle(resp), color='#ff00ff')
        ax3.grid()
        ax3.set_title(f'{label}({r}): phase angle')
        ax3.set_xlim(left=0.0, right=len(signal) - 1)

        idx += 3

    fig.tight_layout()
    plt.show()


def feature_image() -> np.ndarray:
    # Create ideal image with sharp lines and edges.
    ideal = np.zeros((160, 160), dtype=np.float64)

    ideal[:, 19:22] = 1.0
    ideal[:, 60:140] = 1.0
    ideal[:, 99:102] = 0.0

    # Smooth with a gaussian filter.
    image = ndimage.gaussian_filter(ideal, 1.0)

    return image
