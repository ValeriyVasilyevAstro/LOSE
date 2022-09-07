import numpy as np
import emcee
from dataclasses import dataclass
from lightkurve.targetpixelfile import TargetPixelFile
from typing import Callable



@dataclass
class Parameter:
    min: float
    max: float
    value: float
    init_value: float
    err: float


@dataclass
class FittingParameters:
    flare_flux: Parameter
    flare_col: Parameter
    flare_row: Parameter
    offset: Parameter


@dataclass
class MCMCFitting_Input:
    flare_image: np.ndarray
    tpf: TargetPixelFile
    n_steps: int
    n_discard: int


@dataclass
class MCMCFitting_Output:
    chain: np.ndarray
    parameters: FittingParameters
    model: np.ndarray
    data: np.ndarray



class MCMCFitting:
    def run(self, inp: MCMCFitting_Input):
        center_of_brightness = self._initialGuessForCoordinates(target_pixel_data=inp.tpf)
        print("center_of_brightness =", center_of_brightness)
        flare_flux = Parameter(min=0, max=2*np.sum(inp.flare_image[np.invert(np.isnan(inp.flare_image))]),
                               value=None, init_value=np.sum(inp.flare_image[np.invert(np.isnan(inp.flare_image))]) ,
                                                  err = np.sqrt(np.sum(inp.tpf.pipeline_mask)/np.pi)/5 )

        flare_col = Parameter(min=inp.tpf.column, max=inp.flare_image.shape[1] + inp.tpf.column,
                              value=center_of_brightness[1] + inp.tpf.column,
                              init_value=center_of_brightness[1] + inp.tpf.column, err=0.5)
        flare_row = Parameter(min=inp.tpf.row, max=inp.flare_image.shape[0] + inp.tpf.row,
                              value=center_of_brightness[0] + inp.tpf.row,
                              init_value=center_of_brightness[0] + inp.tpf.row, err=0.5)

        offset = Parameter(min=0.0, max=np.max(inp.flare_image[np.invert(np.isnan(inp.flare_image))]),
                           value=np.mean(inp.flare_image[np.invert(np.isnan(inp.flare_image))]),
                           init_value=np.mean(inp.flare_image[np.invert(np.isnan(inp.flare_image))]),
                           err=np.std(inp.flare_image[np.invert(np.isnan(inp.flare_image))]))

        params = FittingParameters(flare_flux=flare_flux, flare_col=flare_col, flare_row=flare_row, offset=offset)
        flat_samples_0, ndim = self._runSampler(flare_image=inp.flare_image, tpf=inp.tpf, params=params,
                                                n_steps=inp.n_steps, n_discard=inp.n_discard)
        p1 = np.mean(flat_samples_0, axis=0)
        params.flare_col.init_value = p1[0]
        params.flare_row.init_value = p1[1]
        params.flare_flux.init_value = p1[2]
        params.offset.init_value = p1[3]
        final_samples, ndim = self._runSampler(flare_image=inp.flare_image, tpf=inp.tpf, params=params,
                                                n_steps=inp.n_steps, n_discard=inp.n_discard)
        final_samples, final_parameters = \
            self._extractBestFittingParametersAndErrorsFromChain(flat_chain_samples=final_samples, ndim=ndim)
        best_fitted_model = self._computeModel(prfmodel=inp.tpf.get_prf_model(),
                                               theta=[final_parameters.flare_col.value,
                                                      final_parameters.flare_row.value,
                                                      final_parameters.flare_flux.value,
                                                      final_parameters.offset.value])
        return MCMCFitting_Output(chain=final_samples, parameters=final_parameters, model=best_fitted_model, data=inp.flare_image)

    def _initialGuessForCoordinates(self, target_pixel_data: TargetPixelFile) -> [float, float]:
        coordinates = np.zeros((1, 2))
        coordinates[0, :] = [target_pixel_data.ra, target_pixel_data.dec]
        pixel_coords = target_pixel_data.wcs.all_world2pix(coordinates, 0)[0]
        return pixel_coords[1], pixel_coords[0]


    def _centerOfBrightnessLocation(self, image: np.ndarray) -> np.ndarray:
        return np.squeeze(np.argwhere(image == image[image == np.nanmax(image)]))

    def _runSampler(self, flare_image: np.ndarray, tpf: TargetPixelFile, params: FittingParameters,
                    n_steps: int, n_discard: int):
        p0 = [params.flare_col.init_value, params.flare_row.init_value,
              params.flare_flux.init_value, params.offset.init_value]
        print("p0:", p0)
        pos = p0 + 1e-2 * np.random.randn(32, len(p0))
        nwalkers, ndim = pos.shape
        prfmodel = tpf.get_prf_model()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._computeLogProbability, args=(flare_image, prfmodel, params))
        sampler.run_mcmc(pos, n_steps)

        chain = sampler.chain[:, n_discard::, :]
        shape = chain.shape
        return chain.reshape(shape[0] * shape[1], shape[2]), ndim

    def _extractBestFittingParametersAndErrorsFromChain(self, flat_chain_samples: np.ndarray, ndim: int) \
            -> [np.ndarray, list, list, list]:
        labels = ["flare col", "flare row", "flare flux", r"offset flux $[e^{-1}s^{-1}]$"]
        flat_chain_samples[:, 0] = flat_chain_samples[:, 0] - 0.5
        flat_chain_samples[:, 1] = flat_chain_samples[:, 1] - 0.5
        for i in range(ndim):
            mcmc = np.percentile(flat_chain_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            print(labels[i], ":", mcmc[1], "-", q[0], "+", q[1])
            par = Parameter(min=mcmc[1] - q[0], max=mcmc[1] + q[1], value=mcmc[1], init_value=None, err=q[0] + q[1])
            if i==0:
                flare_col = par
            if i==1:
                flare_row = par
            if i==2:
                flare_flux = par
            if i==3:
                offset_flux = par
        return flat_chain_samples, FittingParameters(flare_flux=flare_flux, flare_col=flare_col,
                                                     flare_row=flare_row, offset=offset_flux)

    def _computeLogProbability(self, theta: np.ndarray, data: np.ndarray, prfmodel: Callable,
                               par_ranges: FittingParameters) -> float:
        lp = self._computeLogPrior(theta, par_ranges)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._computePoissonLikelihood(theta, data, prfmodel)

    def _computeLogPrior(self, theta: np.ndarray, parameter_ranges: FittingParameters) -> np.ndarray:
        flare_col, flare_row, flare_flux, offset_flux = theta

        if not parameter_ranges.flare_flux.min < flare_flux < parameter_ranges.flare_flux.max:
            return -np.inf
        if not parameter_ranges.flare_col.min < flare_col < parameter_ranges.flare_col.max :
            return -np.inf
        if not parameter_ranges.flare_row.min < flare_row < parameter_ranges.flare_row.max:
            return -np.inf
        if not parameter_ranges.offset.min < offset_flux:
            return -np.inf
        log_prior = 0
        log_prior += np.log(1.0 / (np.sqrt(2 * np.pi) * parameter_ranges.flare_col.err)) - \
                     0.5 * (flare_col - parameter_ranges.flare_col.value) ** 2 / parameter_ranges.flare_col.err ** 2
        log_prior += np.log(1.0 / (np.sqrt(2 * np.pi) * parameter_ranges.flare_row.err)) - \
                     0.5 * (flare_row - parameter_ranges.flare_row.value) ** 2 / parameter_ranges.flare_row.value ** 2
        log_prior += np.log(1.0 / (np.sqrt(2 * np.pi) * parameter_ranges.offset.err)) - \
                     0.5 * (offset_flux - parameter_ranges.offset.value) ** 2 / parameter_ranges.offset.err ** 2
        return log_prior

    def _computePoissonLikelihood(self, theta: np.ndarray, data: np.ndarray, prfmodel: Callable) -> float:
        model_image = self._computeModel(prfmodel, theta)
        return -np.nansum(model_image - data * np.log(model_image))

    def _computeModel(self, prfmodel: Callable, theta: np.ndarray) -> np.ndarray:
        flare_col, flare_row, flare_flux, offset_flux = theta
        predicted_data = prfmodel(center_col=flare_col,
                                  center_row=flare_row,
                                  flux=flare_flux,
                                  scale_col=1,
                                  scale_row=1,
                                  rotation_angle=0.0)
        synthetic_image = predicted_data + offset_flux
        return synthetic_image
