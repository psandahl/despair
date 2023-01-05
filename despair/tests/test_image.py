import despair.image as image
import despair.tests.util as tutil

import numpy as np
import unittest


class ImageTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Setup common stuff before each test function.
        """
        self.src_img = np.arange(1, 10, dtype=np.float64).reshape(3, 3)

        return super().setUp()

    def test_zero_shift(self) -> None:
        """
        Test zero valued shift.
        """
        shift = tutil.global_shift_image(self.src_img.shape, 0.0)
        dst = image.horizontal_shift(self.src_img, shift)
        np.testing.assert_array_almost_equal(dst, self.src_img)

    def test_one_shift(self) -> None:
        """
        Shift one step to the right.
        """
        shift = tutil.global_shift_image(self.src_img.shape, 1.0)

        dst = image.horizontal_shift(self.src_img, shift)
        expected = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8],
                            dtype=np.float64).reshape(3, 3)

        np.testing.assert_array_almost_equal(dst, expected)

    def test_minus_one_shift(self) -> None:
        """
        Shift one step to the left.
        """
        shift = tutil.global_shift_image(self.src_img.shape, -1.0)

        dst = image.horizontal_shift(self.src_img, shift)
        expected = np.array([2, 3, 0, 5, 6, 0, 8, 9, 0],
                            dtype=np.float64).reshape(3, 3)

        np.testing.assert_array_almost_equal(dst, expected)

    def test_half_shift(self) -> None:
        """
        Shift half step to the right.
        """
        shift = tutil.global_shift_image(self.src_img.shape, 0.5)

        dst = image.horizontal_shift(self.src_img, shift)
        expected = np.array([0, 1.5, 2.5, 0, 4.5, 5.5, 0, 7.5, 8.5],
                            dtype=np.float64).reshape(3, 3)

        np.testing.assert_array_almost_equal(dst, expected)

    def test_minus_half_shift(self) -> None:
        """
        Shift half step to the left.
        """
        shift = tutil.global_shift_image(self.src_img.shape, -0.5)

        dst = image.horizontal_shift(self.src_img, shift)
        expected = np.array([1.5, 2.5, 0, 4.5, 5.5, 0, 7.5, 8.5, 0],
                            dtype=np.float64).reshape(3, 3)

        np.testing.assert_array_almost_equal(dst, expected)
