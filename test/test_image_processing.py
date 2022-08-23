import unittest
import numpy as np

from unittest.mock import MagicMock
from src.image_processing import FlareImageComputation


class TestFlareImageComputation(unittest.TestCase):
    def setUp(self):
        self.sut: FlareImageComputation = FlareImageComputation()

    def test_givenTimeArrayAndFlareTimeReturnsTwo(self):
        index_flare = self.sut._find_image_index(flare_time=100.34, time_array=np.array([90.34, 95.34, 100.34, 105.34]))
        self.assertEqual(index_flare, 2)

    def test_givenTimeArrayAndFlareTimeReturnsOne(self):
        index_flare = self.sut._find_image_index(flare_time=95.5, time_array=np.array([90.34, 95.34, 100.34, 105.34]))
        self.assertEqual(index_flare, 1)

