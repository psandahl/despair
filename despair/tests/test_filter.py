import despair.filter as filter

import numpy as np
import unittest


class FilterTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Setup all stuff before each test function.
        """
        self.signal = np.zeros(160, dtype=np.float64)
        self.signal[19:22] = 1.0
        self.signal[60:141] = 1.0
        self.signal[99:102] = 0.0

        self.nonring_filters = list()
        self.wft_filters = list()

        for r in range(3, 10):
            self.nonring_filters.append(filter.nonring(r))
            self.wft_filters.append(filter.wft(r))

        return super().setUp()

    def test_filter_arrays(self):
        """
        Test the filter arrays - and that filters seem constructed ok.
        """
        self.assertEqual(len(self.nonring_filters), 7)
        self.assertEqual(len(self.nonring_filters), len(self.wft_filters))

        r = 3
        for nonring, wft in zip(self.nonring_filters, self.wft_filters):
            self.assertEqual(len(nonring), 2 * r + 1)
            self.assertEqual(len(nonring), len(wft))

            self.assertIsInstance(nonring, np.ndarray)
            self.assertIsInstance(wft, np.ndarray)

            self.assertEqual(nonring.dtype, np.complex128)
            self.assertEqual(wft.dtype, np.complex128)

            r += 1

    def test_nonring_line_response(self):
        """
        Test nonring responses for lines.
        """
        self.__line_response(self.nonring_filters, 'nonring')

    def test_wft_line_response(self):
        """
        Test WFT responses for lines.
        """
        self.__line_response(self.wft_filters, 'wft')

    def __line_response(self, filters: np.ndarray, label: str):
        r = 3
        for filt in filters:
            response = filter.convolve(self.signal, filt)
            magnitude = np.abs(response)
            phase = np.angle(response)
            mean = np.mean(magnitude)

            self.assertGreater(mean, 0.0, msg=f'Filter={label}')

            for index, goal in [(20, 0.0), (100, 3.14159)]:
                mag = magnitude[index]
                ph = phase[index]

                # Magnitude shall be greater than mean. Todo: Find peaks.
                self.assertGreater(
                    mag, mean, msg=f'Filter={label} radius={r}')

                # Phase shall be equal to the goal.
                self.assertAlmostEqual(
                    goal, abs(ph), places=5, msg=f'Filter={label} radius={r}')

            r += 1