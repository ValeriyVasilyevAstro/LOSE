from lightkurve.targetpixelfile import TargetPixelFile
import numpy as np


class Constants:
    cadence: float = 30.0/60/24.0


class FlareTimeIsNotWithinTheTimeRangeInTheTargetPixelFile_Exception(Exception):
    pass


class FlareLocalization:
    def __init__(self, flare_time: float, tpf: TargetPixelFile):
        self._check_that_flare_time_value_is_within_the_time_array_from_tpf(flare_time=flare_time,  time_array=tpf.time.value)
        self.flare_time = flare_time
        self.tpf = tpf
        self.tpf_image_at_flare_time = None
        self.flare_image: np.ndarray = None

    def _check_that_flare_time_value_is_within_the_time_array_from_tpf(self, flare_time: float, time_array: np.ndarray):
        if (flare_time < np.min(time_array)) or (flare_time > np.max(time_array)):
            raise FlareTimeIsNotWithinTheTimeRangeInTheTargetPixelFile_Exception

    def _find_image_index(self,  flare_time: float, time_array: np.ndarray) -> int:
        index_flare = np.squeeze(np.argmin(np.abs(time_array - flare_time)))
        if np.min(np.abs(time_array - flare_time)) != 0:
            print(f"input flare_time {flare_time}  value is not presented in the time array "
                  f"from the target pixel file! Take index corresponding to the nearest value {time_array[index_flare]}")
        return index_flare

    def find_tpf_image_at_flare_time(self) -> "FlareLocalization":
        index = self._find_image_index(flare_time=self.flare_time, time_array=self.tpf_time)
        self.tpf_image_at_flare_time = self.tpf.flux.tpf_images[index]
        return self

    def get_flare_image(self, detrending_method: str = "polynom"):
        """detrend flare"""
        nx = self.tpf_image_at_flare_time.shape[0]
        ny = self.tpf_image_at_flare_time.shape[1]


        return self.flare_image

    def get_image_shape(self) -> [int, int]:



    def localize_flare(self):
        """ """
        return self








