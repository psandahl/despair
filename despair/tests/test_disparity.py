import despair.disparity as disparity
import despair.filter as filter
import despair.image as image
import despair.util as util
import despair.tests.util as tutil

import numpy as np
import scipy.signal as signal
import unittest


class DisparityTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Setup stuff for a simple smoke test. One image, one radius and one shift.
        """

        # Only use one shift scale (the expected disparity will have opposite sign).
        # Negative scale is selected to have a positive disparity (for peak counting).
        # Small disparity for stable testing.
        self.shift_scale = -1.07
        self.radius = 7

        # Setup images.
        reference_img = tutil.feature_image(blur=False)  # No blur.
        shift_img = tutil.global_shift_image(
            reference_img.shape, self.shift_scale)
        query_img = image.horizontal_shift(reference_img, shift_img)

        # Get the filter coefficients for the given radius.
        coeff = filter.coeff(self.radius)

        # Run the disparity computations.
        self.ref_resp = disparity.filter_response(coeff, reference_img)
        self.qry_resp = disparity.filter_response(coeff, query_img)

        self.frequency = disparity.local_frequency(
            self.ref_resp, self.qry_resp)
        self.confidence = disparity.confidence(
            self.ref_resp, self.qry_resp, self.frequency)
        self.phase_difference = disparity.phase_difference(
            self.ref_resp, self.qry_resp)
        self.phase_disparity = disparity.phase_disparity(
            self.phase_difference, self.frequency, self.confidence)

        # Setup convenient signals (image lines).
        self.disparity_signal = self.phase_disparity[0, :]
        self.confidence_signal = self.confidence[0, :]

        return super().setUp()

    def test_signal_arrays(self) -> None:
        """
        Test that images look like what to expect.
        """
        self.assertEqual(self.ref_resp.shape, (160, 160))
        self.assertIsInstance(self.ref_resp, np.ndarray)
        self.assertEqual(self.ref_resp.dtype, np.complex128)

        self.assertEqual(self.qry_resp.shape, (160, 160))
        self.assertIsInstance(self.qry_resp, np.ndarray)
        self.assertEqual(self.qry_resp.dtype, np.complex128)

        self.assertEqual(self.frequency.shape, (160, 160))
        self.assertIsInstance(self.frequency, np.ndarray)
        self.assertEqual(self.frequency.dtype, np.float64)

        self.assertEqual(self.confidence.shape, (160, 160))
        self.assertIsInstance(self.confidence, np.ndarray)
        self.assertEqual(self.confidence.dtype, np.float64)

        self.assertEqual(self.phase_difference.shape, (160, 160))
        self.assertIsInstance(self.phase_difference, np.ndarray)
        self.assertEqual(self.phase_difference.dtype, np.complex128)

        self.assertEqual(self.phase_disparity.shape, (160, 160))
        self.assertIsInstance(self.phase_disparity, np.ndarray)
        self.assertEqual(self.phase_disparity.dtype, np.float64)

    def test_disparity_values(self) -> None:
        """
        Test that there are four disparity peaks, at the expected
        places.
        """

        # Shall be peaks around 20, 60, 100 and 140.
        peaks, _ = signal.find_peaks(self.disparity_signal)
        self.assertEqual(len(peaks), 4)

        for index in [20, 60, 100, 140]:
            self.assertTrue(tutil.is_peak(peaks, index, 1),
                            msg=f'peak@{index}')

            self.assertAlmostEqual(-self.shift_scale,
                                   self.disparity_signal[index],
                                   places=1,
                                   msg=f'value@{index}')

    def test_confidence_values(self) -> None:
        """
        Test that there are four confidence peaks, and
        that confidence is within range.
        """

        self.assertGreaterEqual(np.min(self.confidence_signal), 0.0)
        self.assertLessEqual(np.max(self.confidence_signal), 1.0)

        # Shall be peaks around 20, 60, 100 and 140.
        peaks, _ = signal.find_peaks(self.confidence_signal)
        self.assertEqual(len(peaks), 4)

        for index in [20, 60, 100, 140]:
            self.assertTrue(tutil.is_peak(peaks, index, 1),
                            msg=f'peak@{index}')
