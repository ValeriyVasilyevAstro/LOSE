import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class FlareImageComputation_Input:
    images: np.ndarray
    time: np.ndarray
    cadences_all: np.ndarray
    cadences_of_flare: np.ndarray


@dataclass
class FlareImageComputation_Output:
    variances: np.ndarray
    mean_fluxes: np.ndarray
    mean_flux_flare: np.ndarray
    flare_image: np.ndarray
    all_images: np.ndarray
    time: np.ndarray


class ProcessingFlareImageComputation:
    def processing(self, inp: FlareImageComputation_Input) -> FlareImageComputation_Output:
        cadences_before_and_after_the_flare = self._getCadencesBeforeAndAfterTheFlare(inp.cadences_of_flare,
                                                                                      inp.cadences_all)
        n_xpix = inp.images[cadences_before_and_after_the_flare][0, :, :].shape[0]
        n_ypix = inp.images[cadences_before_and_after_the_flare][0, :, :].shape[1]
        variance = np.zeros((n_xpix, n_ypix))
        quiet_stellar_flux_cadences = np.zeros((len(inp.cadences), n_xpix, n_ypix))
        mean_flux_flare = np.zeros((1, n_xpix, n_ypix))
        print("time stamp for image computation:", inp.time[inp.cadence_flare])
        for i in range(n_xpix):
            for j in range(n_ypix):
                if np.isnan(inp.images[inp.cadences][0, i, j]):
                    variance[i, j] = np.nan
                else:
                    quiet_stellar_flux = self._fitPolynom(x=inp.time[inp.cadences_before_and_after_the_flare],
                                                          y=inp.images[inp.cadences_before_and_after_the_flare][:, i, j],
                                                          order=3)
                    variance[i, j] = self._computeStd(data_flux=inp.images[inp.cadences_before_and_after_the_flare][:, i, j],
                                                      mean_flux=quiet_stellar_flux(inp.time[inp.cadences_before_and_after_the_flare]))
                    quiet_stellar_flux_cadences[:, i, j] = quiet_stellar_flux(inp.time[inp.cadences])
                    mean_flux_flare[:, i, j] = quiet_stellar_flux(inp.time[inp.cadence_flare])
        residual_flare_flux = inp.images[inp.cadence_flare] - mean_flux_flare[0, :, :]
        residual_flare_flux = residual_flare_flux + abs(np.min(residual_flare_flux[np.invert(np.isnan(residual_flare_flux))]))
        return FlareImageComputation_Output(variances=variance,
                                            mean_fluxes=quiet_stellar_flux_cadences,
                                            mean_flux_flare=mean_flux_flare,
                                            flare_image=residual_flare_flux,
                                            all_images = inp.images[inp.cadences],
                                            time = inp.time[inp.cadences])

    def _getCadencesBeforeAndAfterTheFlare(self, cadences_of_flare: np.ndarray, cadences: np.ndarray) -> np.ndarray:
        cadence_flare = np.array(np.argmin(abs(time - time_flare)))
        cadence_around_flare_with_flare = np.linspace(cadence_flare - 33, cadence_flare + 33, 67, dtype=int)
        cadence_around_flare_witout_flare = np.concatenate(
            (np.linspace(cadence_flare - 33, cadence_flare - 2, 32, dtype=int),
             np.linspace(cadence_flare + 4, cadence_flare + 33, 30, dtype=int)), axis=0)
        if len(time) <= np.max(cadence_around_flare_with_flare):
            indx_max = np.argwhere(cadence_around_flare_with_flare == len(time))[0]
            cadence_around_flare_with_flare = cadence_around_flare_with_flare[0:indx_max[0]]
            indx_max = np.argwhere(cadence_around_flare_witout_flare == len(time))[0]
            cadence_around_flare_witout_flare = cadence_around_flare_witout_flare[0:indx_max[0]]
        return
