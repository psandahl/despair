import matplotlib.pyplot as plt
import numpy as np

import despair.filter as filter


def plot_with_radius(r: float) -> None:
    """
    Plot the filters using the given radius.

    Parameters:
        r: Radius of the filter.
    """
    x = np.arange(-r, r + 1, dtype=np.float64)

    filters = [
        ("Nonring", filter.nonring)
    ]

    fig = plt.figure(figsize=(8, 2 * len(filters)))

    idx = 1
    for label, func in filters:
        filt = func(r)

        ax1 = fig.add_subplot(1, 3, idx)
        ax1.plot(x, filt.real, color='#0000ff',
                 linewidth=2)
        ax1.set_title(f'{label} real part')

        ax2 = fig.add_subplot(1, 3, idx + 1)
        ax2.plot(x, filt.imag, color='#00ff00',
                 linewidth=2)
        ax2.set_title(f'{label} imaginary part')

        ax3 = fig.add_subplot(1, 3, idx + 2)
        ax3.plot(x, filt.real, color='#0000ff',
                 linewidth=2)
        ax3.plot(x, filt.imag, color='#00ff00',
                 linewidth=2)
        ax3.set_title(f'{label} complex')

        idx += 3

    fig.suptitle(f'Filters with radius={r}')
    fig.tight_layout()
    plt.show()
