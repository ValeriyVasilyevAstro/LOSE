import sys
import os
from astropy.io import fits
import numpy as np
import copy
import pandas
from dataclasses import dataclass
from lightkurve.targetpixelfile import TargetPixelFile
import lightkurve as lk
from confidence_ellipce_estimation import ConfidenceEllipceEstimation_Input, ConfidenceEllipceEstimation, FlareEllipce
from flare_image_computation import FlareImageComputation_Input, FlareImageComputation_Output, \
    ProcessingFlareImageComputation
from tpf_selection import TargetPixelDataPreparation_Input, TargetPixelDataPreparation_Output, \
    TargetPixelDataPreparation, FlareParameters
from mcmc import MCMCFitting, MCMCFitting_Input, FittingParameters



class ExceptionDataNone(Exception):
    pass


central_folder = "/data/seismo/vasilyev/flares/"
folder_with_tpfs = central_folder+"tpf/"
folder_to_load_gaia = central_folder+"gaia/"
main_folder_to_save = "/data/seismo/vasilyev/flares/results/solar_like_stars/"
main_folder_to_save_mcmc_fitting = main_folder_to_save + "mcmc_result/"
main_folder_to_save_detrended_fluxes = main_folder_to_save + "plots/tpf_detrending/"
main_folder_to_save_mcmc_plots = main_folder_to_save + "plots/mcmc/"
make_plot_key = True
folder_light_curves = "/data/seismo/vasilyev/flares/results/solar_like_stars/stds_and_smoothed_fluxes/"

list_of_flare_candidates = np.load("/data/seismo/vasilyev/flares/results/solar_like_stars/flare_candidates_light_curve_analysis.npy", allow_pickle=True).item()

import sys
star_star_number = int(sys.argv[1])


kic_id= int(list_of_flare_candidates["kic_id"][star_star_number])
quarter = int(list_of_flare_candidates["q"][star_star_number])
time_1 = list_of_flare_candidates["time"][star_star_number]
cadence_1 = list_of_flare_candidates["cad"][star_star_number]

time_2 =  list_of_flare_candidates["time_next"][star_star_number]
cadence_2 = list_of_flare_candidates["cad_next"][star_star_number]






def loadGAIA(tpf: TargetPixelFile, kik_id: int, folder_to_load_gaia: str) -> [np.ndarray, np.ndarray]:
    print("download GAIA data from", "kic_id"+str(kik_id)+".csv")
    data = pandas.read_csv(folder_to_load_gaia+"kic_id"+str(kik_id)+".csv")
    radecs_2015 = np.vstack([data['RA_ICRS'], data['DE_ICRS']]).T
    ra_dec_pix_2015 = tpf.wcs.all_world2pix(radecs_2015, 0)
    ra_dec_pix_2015[:,0] = ra_dec_pix_2015[:, 0] + tpf.column
    ra_dec_pix_2015[:,1] = ra_dec_pix_2015[:, 1] + tpf.row
    return ra_dec_pix_2015, data



@dataclass
class InputForFitting:
    flare_parameters: FlareParameters
    half_window_around_flare: float = 0.3
    n_steps: int = 1200#0
    n_discard: int = 800#0


class Processing:

    def run(self, inp: InputForFitting):
        tpfs: TargetPixelDataPreparation_Output = TargetPixelDataPreparation().run(
            inp=TargetPixelDataPreparation_Input(flare_parameters=inp.flare_parameters,
                                                 folder_with_tpd_files = folder_with_tpfs,
                                                 half_window_around_flare = inp.half_window_around_flare))

        image_with_flare: FlareImageComputation_Output = ProcessingFlareImageComputation().processing(
            inp=FlareImageComputation_Input(images=tpfs.target_pixel_data.flux.value,
                                        time=tpfs.target_pixel_data.time.value,
                                        cadences_before_and_after_the_flare=tpfs.cadence_around_flare_without_flare,
                                        cadence_flare=tpfs.cadence_flare,
                                        cadences=tpfs.cadences))
        print("image_with_flare.all_images:", image_with_flare.all_images.shape)
        print("tpfs.target_pixel_data.pipeline_mask:", tpfs.target_pixel_data.pipeline_mask,
              np.sqrt(np.sum(tpfs.target_pixel_data.pipeline_mask)/np.pi)/5)


        if np.count_nonzero(~np.isnan(image_with_flare.flare_image)) == 0:
            print("error in main: np.count_nonzero(~np.isnan(image_with_flare.flare_image)) == 0 !!!!!")
            raise ExceptionDataNone
        else:
            ra_dec_pix_2015, data_gaia = loadGAIA(tpf=tpfs.target_pixel_data, kik_id=inp.flare_parameters.kic_id,
                                                  folder_to_load_gaia=folder_to_load_gaia)

            mcmc_fitting = MCMCFitting().run(inp=MCMCFitting_Input(flare_image=image_with_flare.flare_image,
                                                                   tpf=tpfs.target_pixel_data, n_steps=inp.n_steps,
                                                                   n_discard=inp.n_discard))
            conf_ell = ConfidenceEllipceEstimation().run(inp=ConfidenceEllipceEstimation_Input(mcmc_chain=mcmc_fitting.chain,
                                                                                               data_gaia=data_gaia,
                                                                                               ra_dec_in_pix=ra_dec_pix_2015))
            quality_flag_for_flare = tpfs.target_pixel_data.quality[tpfs.cadence_flare]
            data_to_be_saved = {"quality_flag_nr": quality_flag_for_flare,
                                "flare_cadence": tpfs.cadence_flare,
                                "cadences_near_flare": tpfs.cadences,
                                "quality_flag": lk.KeplerQualityFlags.decode( quality_flag_for_flare ),
                                "chain":mcmc_fitting.chain, "data": mcmc_fitting.data,
                                "data_gaia":data_gaia, "model": mcmc_fitting.model, "result": mcmc_fitting.parameters,
                                "stars_within_99pes_or_nearest": conf_ell.stars_near_the_99pers_or_nearest,
                                "quiet_pixel_fluxes": image_with_flare.mean_fluxes,
                                "quiet_pixel_flux_flare": image_with_flare.mean_flux_flare,
                                "all_pixel_fluxes": image_with_flare.all_images,
                                "time": image_with_flare.time}
            np.save(main_folder_to_save_mcmc_fitting+"kic_id"+str(inp.flare_parameters.kic_id)+
                    "_q"+str(inp.flare_parameters.quarter)+"_t"+str(round(inp.flare_parameters.time,3))+
                    "_mcmc_chain.npy", data_to_be_saved)
            return data_to_be_saved, conf_ell, tpfs, ra_dec_pix_2015, data_gaia

##################################################################################################

processing = Processing()
try:
    result_cadence_1, conf_ell_1, tpf_1, ra_dec_pix_2015, data_gaia = \
        processing.run(inp=InputForFitting(FlareParameters(kic_id=kic_id, quarter=quarter, time=time_1, cadence=cadence_1)))
except ExceptionDataNone:
    pass


