import numpy as np
from dataclasses import dataclass
import glob
import lightkurve as lk
from lightkurve.targetpixelfile import TargetPixelFile


@dataclass
class FlareParameters:
    kic_id: int
    quarter: int
    time: float
    cadence: int

@dataclass
class TargetPixelDataPreparation_Input:
    flare_parameters: FlareParameters
    folder_with_tpd_files: str
    half_window_around_flare: float = 0.3


@dataclass
class TargetPixelDataPreparation_Output:
    target_pixel_data: TargetPixelFile
    cadence_flare: np.ndarray
    cadences: np.ndarray
    cadence_around_flare_without_flare: np.ndarray


class NoTargetPixelFile(Exception):
    pass


class TargetPixelDataPreparation:
    def run(self, inp: TargetPixelDataPreparation_Input) -> TargetPixelDataPreparation_Output:
        print("Run TargetPixelDataPreparation for target: ", inp.flare_parameters.kic_id, " q:",
              inp.flare_parameters.quarter, " t_flare:", inp.flare_parameters.time)
        target_pixel_data = self._getTargetPixelData(folder_to_tpd=inp.folder_with_tpd_files,
                                                     kik_id=inp.flare_parameters.kic_id,
                                                     quarter=inp.flare_parameters.quarter)
        target_pixel_data = self._filterTargetPixelData(target_pixel_datas=target_pixel_data,
                                                        time_flare=inp.flare_parameters.time)
        cadence_flare, cadence_around_flare_with_flare, cadence_around_flare_without_flare = \
            self._findIndicesOfCadencesAroundFlare(time=target_pixel_data.time.value, time_flare=inp.flare_parameters.time)
        return TargetPixelDataPreparation_Output(target_pixel_data=target_pixel_data,
                                                 cadence_flare=cadence_flare,
                                                 cadences=cadence_around_flare_with_flare,
                                                 cadence_around_flare_without_flare=cadence_around_flare_without_flare)

    def _getTargetPixelData(self, folder_to_tpd: str, kik_id: int, quarter: int) -> list:
        filename = "kic_id"+str(kik_id)+"_Q"+str(quarter)+"*.fits"
        print("folder_to_tpd:", folder_to_tpd)
        print("filename:", filename)

        all_target_pixel_files_for_given_quarter = glob.glob(folder_to_tpd + filename)
        target_pixel_datas = []
        print(all_target_pixel_files_for_given_quarter)
        if len(all_target_pixel_files_for_given_quarter)==0:
            print("error in _getTargetPixelData:  No TPD file for kic_id ", kik_id, "  q", quarter)
            raise NoTargetPixelFile
        for f in all_target_pixel_files_for_given_quarter:
                target_pixel_datas.append(lk.read(f))
        return target_pixel_datas

    def _filterTargetPixelData(self, target_pixel_datas: list, time_flare: float) -> TargetPixelFile:
        if len(target_pixel_datas) > 1:
            print("_filterTargetPixelData:  Warning  more than one file")
            for i in range(len(target_pixel_datas)):
                time = target_pixel_datas[i].time.value
                time = time[~np.isnan(time)]
                if (np.min(time) < time_flare) & (np.max(time) > time_flare):
                    print("_filterTargetPixelData: file is selected")
                    return target_pixel_datas[i]
                else:
                    print("error in _filterTargetPixelData: no target pixel data")
                    raise NoTargetPixelFile
        return target_pixel_datas[0]

    def _findIndicesOfCadencesAroundFlare(self, time, time_flare) -> [np.ndarray, np.ndarray, np.ndarray]:
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
        return cadence_flare, cadence_around_flare_with_flare, cadence_around_flare_witout_flare



