from lightkurve.targetpixelfile import TargetPixelFile
import numpy as np


class Constants:
    cadence: float = 30.0/60/24.0
    number_of_cadences_in_one_hour_lc: int = 2


class Parameters:
    number_of_excluded_cadences_before_flare: int = 2
    number_of_excluded_cadences_after_flare: int = 4


class FlareTimeIsNotWithinTheTimeRangeInTheTargetPixelFile_Exception(Exception):
    pass


class FlareLocalization:
    def __init__(self, flare_time: float, tpf: TargetPixelFile, window_length_hours: float = 16.5):
        self._check_that_flare_time_value_is_within_the_time_array_from_tpf(flare_time=flare_time,  time_array=tpf.time.value)
        self.flare_time = flare_time
        self.tpf = tpf
        self.window_length_hours = window_length_hours
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
        index_flare = self._find_image_index(flare_time=self.flare_time, time_array=self.tpf.time.value)
        self.tpf_image_at_flare_time = self.tpf.flux.value[index_flare]
        return self

    def get_flare_image(self, detrending_method: str = "polynom"):
        """detrend flare"""
        nx = self.tpf_image_at_flare_time.shape[0]
        ny = self.tpf_image_at_flare_time.shape[1]
        index_flare = self._find_image_index(flare_time=self.flare_time, time_array=self.tpf.time.value)
        window_length = int(self.window_length_hours * Constants.number_of_cadences_in_one_hour_lc)
        indices_of_window_with_flare = np.linspace(index_flare - window_length,
                                                          index_flare + window_length,
                                                          int(self.window_length_hours)+1, dtype=int)
        indices_of_window_with_exlcuded_flare = np.concatenate(
            (np.linspace(index_flare - window_length,
                         index_flare - Parameters.number_of_excluded_cadences_before_flare,
                         window_length - 1, dtype=int),
             np.linspace(index_flare + Parameters.number_of_excluded_cadences_after_flare,
                         index_flare + window_length, window_length-3, dtype=int)), axis=0)

        return self.flare_image

    def localize_flare(self):
        """ """
        return self








