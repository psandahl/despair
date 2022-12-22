import matplotlib.pyplot as plt
import numpy as np

import despair.filter as filter


def filters_with_radius(r: float) -> None:
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
        ax1.set_title(f'{label} re/even part')

        ax2 = fig.add_subplot(num_filters, 3, idx + 1)
        ax2.plot(x, filt.imag, color='#00ff00',
                 linewidth=2)
        ax2.set_title(f'{label} im/odd part')

        ax3 = fig.add_subplot(num_filters, 3, idx + 2)
        ax3.plot(x, filt.real, color='#0000ff',
                 linewidth=2)
        ax3.plot(x, filt.imag, color='#00ff00',
                 linewidth=2)
        ax3.set_title(f'{label} complex')

        idx += 3

    fig.suptitle(f'Filters with radius={r}')
    fig.tight_layout()
    plt.show()


def responses_with_radius(r: float) -> None:
    """
    Plot the filter responses using the given radius.

    Parameters:
        r: Radius of the filter.
    """
    assert r > 0

    size = r * 3
    x = np.arange(size, dtype=np.float64)

    features = [
        ("line", make_line),
        ("up edge", make_up_edge),
        ("down edge", make_down_edge)
    ]
    num_features = len(features)

    # Assume these two filters at the moment.
    nonring = filter.nonring(r)
    wft = filter.wft(r)

    fig = plt.figure(figsize=(8, num_features * 2))

    idx = 1
    for label, func in features:
        feature = func(size)

        ax1 = fig.add_subplot(num_features, 3, idx)
        ax1.plot(x, feature, color='#ffff00')
        ax1.set_title(f'{label}')

        resp_nonring = filter.convolve(nonring, feature)
        ax2 = fig.add_subplot(num_features, 3, idx + 1)
        ax2.plot(x, resp_nonring.real, color='#0000ff')
        ax2.plot(x, resp_nonring.imag, color='#00ff00')
        ax2.set_title('Response nonring filter')

        resp_wft = filter.convolve(wft, feature)
        ax3 = fig.add_subplot(num_features, 3, idx + 2)
        ax3.plot(x, resp_wft.real, color='#0000ff')
        ax3.plot(x, resp_wft.imag, color='#00ff00')
        ax3.set_title('Response WFT filter')

        idx += 3

    fig.tight_layout()
    plt.show()


def make_line(size: float) -> np.ndarray:
    line = np.zeros(size, dtype=np.float64)

    mid = size // 2
    line[mid - 2] = 0.25
    line[mid - 1] = 0.75
    line[mid] = 0.9
    line[mid + 1] = 0.75
    line[mid + 2] = 0.25

    return line


def make_up_edge(size: float) -> np.ndarray:
    edge = np.zeros(size, dtype=np.float64)

    mid = size // 2

    edge[:mid] = 0.25
    edge[mid - 1] = 0.5
    edge[mid:] = 0.75

    return edge


def make_down_edge(size: float) -> np.ndarray:
    edge = np.zeros(size, dtype=np.float64)

    mid = size // 2

    edge[:mid] = 0.75
    edge[mid - 1] = 0.5
    edge[mid:] = 0.25

    return edge
