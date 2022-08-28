import numpy as np
from lightkurve.targetpixelfile import TargetPixelFile
from dataclasses import dataclass
from typing import Callable
from src.config import Constants, Parameters


class FlareTimeIsNotWithinTheTimeRangeInTheTargetPixelFile_Exception(Exception):
    pass


@dataclass
class FlareImageComputation_Input:
    tpf: TargetPixelFile
    flare_time: float
    window_length_hours: float = Parameters.window_length_hours


@dataclass
class FlareImageComputation_Output:
    tpf_image_at_flare_time: np.ndarray
    quiet_stellar_fluxes_within_window_with_flare: np.ndarray
    quiet_stellar_flux_flare_cadence: np.ndarray
    flare_image: np.ndarray
    tpfs_within_window_with_flare: np.ndarray
    time_within_window_with_flare: np.ndarray



class FlareImageComputation:
    def processing(self, inp=FlareImageComputation_Input) -> FlareImageComputation_Output:
        window_length = int(inp.window_length_hours * Constants.number_of_cadences_in_one_hour_lc)
        index_flare = self._find_image_index(flare_time=inp.flare_time, time_array=inp.tpf.time.value)
        tpf_image_at_flare_time = inp.tpf.flux.value[index_flare]
        flare_image, quiet_stellar_flux_at_flare_cadence, quiet_stellar_fluxes_within_window_with_flare, \
            tpfs_within_window_with_flare, time_within_window_with_flare = self._compute_flare_image(index_flare=index_flare,
                                                                      images=inp.tpf.flux.value,
                                                                      time=inp.tpf.time.value,
                                                                      window_length=window_length)
        quiet_stellar_flux_flare_cadence = quiet_stellar_flux_at_flare_cadence
        return FlareImageComputation_Output(tpf_image_at_flare_time,
                                            quiet_stellar_fluxes_within_window_with_flare,
                                            quiet_stellar_flux_flare_cadence,
                                            flare_image,
                                            tpfs_within_window_with_flare,
                                            time_within_window_with_flare)

    def _find_image_index(self,  flare_time: float, time_array: np.ndarray) -> int:
        index_flare = np.squeeze(np.argmin(np.abs(time_array - flare_time)))
        if np.min(np.abs(time_array - flare_time)) != 0:
            print(f"WARNING: input flare_time {flare_time}  value is not presented in the time array "
                  f"from the target pixel file! Take index corresponding to the nearest value {time_array[index_flare]}")
        return index_flare

    def _compute_flare_image(self, index_flare: int, images: np.ndarray, time: np.ndarray, window_length: int):
        """detrend flare"""
        indices_within_window_with_flare = self._get_indices_within_window_including_flare(index_flare, window_length)
        indices_within_window_without_flare = self._get_indices_within_window_without_flare_cadence(index_flare, window_length)
        nx = images[index_flare].shape[0]
        ny = images[index_flare].shape[1]
        quiet_stellar_flux_within_window_with_flare = np.zeros((len(indices_within_window_with_flare), nx, ny))
        quiet_stellar_flux_at_flare_cadence = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                if not np.isnan(images[indices_within_window_with_flare][0, i, j]):
                    polynom_quiet_stellar_flux = self._fit_polynom(x=time[indices_within_window_without_flare],
                                                                   y=images[indices_within_window_without_flare][:, i, j],
                                                                   order=Parameters.order_of_the_fitted_polynomial)
                    quiet_stellar_flux_within_window_with_flare[:, i, j] = polynom_quiet_stellar_flux(time[indices_within_window_with_flare])
                    quiet_stellar_flux_at_flare_cadence[i, j] = polynom_quiet_stellar_flux(time[index_flare])
        flare_image = self._remove_quite_stellar_flux_from_tpf(images[index_flare], quiet_stellar_flux_at_flare_cadence)
        flare_image = flare_image + self._compute_offset(flare_image)
        tpfs_within_window_with_flare = images[indices_within_window_with_flare]
        time_within_window_with_flare = time[indices_within_window_with_flare]
        return flare_image, quiet_stellar_flux_at_flare_cadence, quiet_stellar_flux_within_window_with_flare, \
            tpfs_within_window_with_flare, time_within_window_with_flare

    def _remove_quite_stellar_flux_from_tpf(self, tpf_at_flare_cadence, quiet_stellar_flux_at_flare_cadence):
        return tpf_at_flare_cadence - quiet_stellar_flux_at_flare_cadence

    def _compute_offset(self, residual_flare_flux: np.ndarray):
        """the offset is introduced to avoid negative pixel counts and later taken into account during fitting"""
        return abs(np.min(residual_flare_flux[np.invert(np.isnan(residual_flare_flux))]))

    def _get_indices_within_window_without_flare_cadence(self, index_flare: int, window_length: int) -> np.ndarray:
        indices_within_window_with_exlcuded_flare = np.concatenate(
            (np.linspace(index_flare - window_length,
                         index_flare - Parameters.number_of_excluded_cadences_before_flare,
                         window_length - 1, dtype=int),
             np.linspace(index_flare + Parameters.number_of_excluded_cadences_after_flare,
                         index_flare + window_length, window_length - 3, dtype=int)), axis=0)
        return indices_within_window_with_exlcuded_flare

    def _get_indices_within_window_including_flare(self, index_flare: int, window_length: int) -> np.ndarray:
        return np.linspace(index_flare - window_length, index_flare + window_length, int(window_length) + 1, dtype=int)

    def _fit_polynom(self, x: np.ndarray, y: np.ndarray, order: int) -> Callable:
        z = np.polyfit(x, y, order)
        return np.poly1d(z)

    def _compute_std(self, data_flux: np.ndarray, mean_flux: np.ndarray) -> np.ndarray:
        return np.std(data_flux - mean_flux)
