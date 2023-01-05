import despair.filter as filter
import despair.util as util
import despair.tests.util as tutil

import math
import numpy as np
import scipy.signal as signal
import unittest


def is_peak(peaks: np.ndarray, target: int, margin: int = 0) -> bool:
    return np.min(np.abs(peaks - target)) <= margin


class FilterTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Setup all stuff before each test function.
        """
        self.signal = tutil.feature_image()[0, :]

        self.filters = list()

        for r in range(3, 10):
            self.filters.append(filter.coeff(r))

        return super().setUp()

    def test_filter_arrays(self) -> None:
        """
        Test the filter arrays - and that filters seem constructed ok.
        """
        self.assertEqual(len(self.filters), 7)

        r = 3
        for coeff in self.filters:
            self.assertEqual(len(coeff), 2 * r + 1)
            self.assertIsInstance(coeff, np.ndarray)
            self.assertEqual(coeff.dtype, np.complex128)

            r += 1

    def test_line_response(self) -> None:
        """
        Test filter response for lines.
        """
        r = 3
        for coeff in self.filters:
            response = filter.convolve(self.signal, coeff)
            magnitude = np.abs(response)
            phase = np.angle(response)
            mean = np.mean(magnitude)

            self.assertGreater(mean, 0.0)

            # Get the magnitude peaks.
            peaks, _ = signal.find_peaks(magnitude)

            # Hack warning ...
            self.assertLessEqual(len(peaks), 6)

            for index, goal in [(20, 0.0), (100, 3.14159)]:
                mag = magnitude[index]
                ph = phase[index]

                # The target index shall be a peak in the magnitude.
                self.assertTrue(is_peak(peaks, index, 2), msg=f'radius={r}')

                # Phase shall be equal to the goal.
                self.assertAlmostEqual(
                    goal, abs(ph), places=5, msg=f'radius={r}')

            r += 1

    def test_edge_response(self) -> None:
        """
        Test filter response for edges.
        """
        r = 3
        for coeff in self.filters:
            response = filter.convolve(self.signal, coeff)

            magnitude = np.abs(response)
            phase = np.angle(response)
            mean = np.mean(magnitude)

            self.assertGreater(mean, 0.0)

            # Get the magnitude peaks.
            peaks, _ = signal.find_peaks(magnitude)

            # Hack warning ...
            self.assertLessEqual(len(peaks), 6)

            for index0, index1, goal in [(59, 60, math.pi / 2), (140, 141, -math.pi / 2)]:
                # For edges in the test signal the exact response lies in between
                # the pixels, and has to be interpolated.
                mag = util.mix(magnitude[index0], magnitude[index1], 0.5)
                ph = util.mix(phase[index0], phase[index1], 0.5)

                # The target index shall be a peak in the magnitude.
                self.assertTrue(is_peak(peaks, index0, 1), msg=f'radius={r}')

                # Phase shall be equal to the goal.
                self.assertAlmostEqual(goal, ph, places=2)

            r += 1
