from lightkurve.targetpixelfile import TargetPixelFile
import numpy as np
from typing import Callable


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
        self.window_length = int(window_length_hours * Constants.number_of_cadences_in_one_hour_lc)
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
        index_flare = self._find_image_index(flare_time=self.flare_time, time_array=self.tpf.time.value)
        indices_within_window_with_flare = self._get_indices_within_window_including_flare(index_flare)
        indices_within_window_without_flare = self._get_indices_within_window_without_flare_cadence(index_flare)
        nx = self.tpf_image_at_flare_time.shape[0]
        ny = self.tpf_image_at_flare_time.shape[1]

        images = self.tpf.flux.value
        time = self.tpf.time.value
        variance = np.zeros((nx, ny))
        quiet_stellar_flux_cadences = np.zeros((len(indices_within_window_with_flare), nx, ny))
        quiet_stellar_flux_at_flare_cadence = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                if np.isnan(images[indices_within_window_with_flare][0, i, j]):
                    variance[i, j] = np.nan
                else:
                    polynom_quiet_stellar_flux = self._fit_polynom(x=time[indices_within_window_without_flare],
                                                                   y=images[indices_within_window_without_flare][:, i, j],
                                                                   order=3)
                    variance[i, j] = self._compute_std(
                        data_flux=images[indices_within_window_without_flare][:, i, j],
                        mean_flux=polynom_quiet_stellar_flux(time[indices_within_window_without_flare]))
                    quiet_stellar_flux_cadences[:, i, j] = polynom_quiet_stellar_flux(time[indices_within_window_with_flare])
                    quiet_stellar_flux_at_flare_cadence[i, j] = polynom_quiet_stellar_flux(time[index_flare])
        residual_flare_flux = self._remove_quite_stellar_flux_from_tpf(images[indices_within_window_with_flare], quiet_stellar_flux_at_flare_cadence)
        self.flare_image = residual_flare_flux + self._compute_offset(residual_flare_flux)
        return self


    def _remove_quite_stellar_flux_from_tpf(self, tpf_at_flare_cadence, quiet_stellar_flux_at_flare_cadence):
        return tpf_at_flare_cadence - quiet_stellar_flux_at_flare_cadence

    def _compute_offset(self, residual_flare_flux: np.ndarray):
        """the offset is introduced to avoid negative pixel counts and later taken into account during fitting"""
        return abs(np.min(residual_flare_flux[np.invert(np.isnan(residual_flare_flux))]))

    def _get_indices_within_window_without_flare_cadence(self, index_flare: int) -> np.ndarray:
        indices_within_window_with_exlcuded_flare = np.concatenate(
            (np.linspace(index_flare - self.window_length,
                         index_flare - Parameters.number_of_excluded_cadences_before_flare,
                         self.window_length - 1, dtype=int),
             np.linspace(index_flare + Parameters.number_of_excluded_cadences_after_flare,
                         index_flare + self.window_length, self.window_length - 3, dtype=int)), axis=0)
        return indices_within_window_with_exlcuded_flare

    def _get_indices_within_window_including_flare(self, index_flare: int) -> np.ndarray:
        return np.linspace(index_flare - self.window_length, index_flare + self.window_length, int(self.window_length) + 1, dtype=int)

    def _fit_polynom(self, x: np.ndarray, y: np.ndarray, order: int) -> Callable:
        z = np.polyfit(x, y, order)
        return np.poly1d(z)

    def _compute_std(self, data_flux: np.ndarray, mean_flux: np.ndarray) -> np.ndarray:
        return np.std(data_flux - mean_flux)









