import despair.filter as filter
import despair.util as util

import numpy as np
import unittest


class FilterTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Setup all stuff before each test function.
        """
        self.signal = util.feature_image()[0, :]

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
