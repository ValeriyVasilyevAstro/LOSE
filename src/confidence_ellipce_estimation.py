import numpy as np
from matplotlib.patches import Ellipse
from dataclasses import dataclass


default_percentiles = np.array([0.68, 0.95, 0.999])
default_chi2_values = np.array([2.27887, 5.99146, 13.81551])

@dataclass
class FlareEllipce:
    ellipce_plot: Ellipse
    width: float
    height: float
    center_xy: np.ndarray
    theta: float


@dataclass
class ConfidenceEllipceEstimation_Input:
    mcmc_chain: np.ndarray
    data_gaia: dict
    ra_dec_in_pix: np.ndarray
    percentiles: np.ndarray = default_percentiles
    chi2_values: np.ndarray = default_chi2_values


@dataclass
class ConfidenceEllipceEstimation_Output:
    flare_ell_68: FlareEllipce
    flare_ell_95: FlareEllipce
    flare_ell_99: FlareEllipce
    stars_near_the_99pers_or_nearest: dict


class ConfidenceEllipceEstimation:
    def run(self, inp: ConfidenceEllipceEstimation_Input) -> ConfidenceEllipceEstimation_Output:
        flare_ellipce_68  = self._generateEllipse(x=inp.mcmc_chain[:, 0], y=inp.mcmc_chain[:, 1],
                                                  persentage=default_percentiles[0])
        flare_ellipce_95  = self._generateEllipse(x=inp.mcmc_chain[:, 0], y=inp.mcmc_chain[:, 1],
                                                  persentage=default_percentiles[1])
        flare_ellipce_999 = self._generateEllipse(x=inp.mcmc_chain[:, 0], y=inp.mcmc_chain[:, 1],
                                                  persentage=default_percentiles[2])
        stars_near_the_99pers_or_nearest = self._checkIsStarInTheErrorEllipse(data_gaia=inp.data_gaia,
                                                                              flare_ellipce=flare_ellipce_999,
                                                                              ra_dec_in_pix=inp.ra_dec_in_pix)
        return ConfidenceEllipceEstimation_Output(flare_ell_68=flare_ellipce_68,
                                                  flare_ell_95=flare_ellipce_95,
                                                  flare_ell_99=flare_ellipce_999,
                                                  stars_near_the_99pers_or_nearest=stars_near_the_99pers_or_nearest)

    def _eigsorted(self, cov: np.ndarray) -> [np.ndarray, np.ndarray]:
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    def _generateEllipse(self, x: np.ndarray,y: np.ndarray, persentage: float) -> FlareEllipce:
        cov = np.cov(x, y)
        vals, vecs = self._eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        if persentage == default_percentiles[0]:
            nstd = np.sqrt(default_chi2_values[0])
        if persentage == default_percentiles[1]:
            nstd = np.sqrt(default_chi2_values[1])
        if persentage == default_percentiles[2]:
            nstd = np.sqrt(default_chi2_values[2])
        w, h = 2 * nstd * np.sqrt(vals)
        ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=theta, color='red')
        return FlareEllipce(ellipce_plot=ell,  width=w,  height=h,  center_xy=np.array([np.mean(x), np.mean(y)]), theta=theta)




    def _checkIsStarInTheErrorEllipse(self, data_gaia: dict, flare_ellipce: FlareEllipce, ra_dec_in_pix: np.ndarray,
                                      persentage: float=0.999) -> dict:
        cos_angle = np.cos(np.radians(180.0-flare_ellipce.theta))
        sin_angle = np.sin(np.radians(180.0-flare_ellipce.theta))
        x = ra_dec_in_pix[:,0]
        y = ra_dec_in_pix[:,1]
        xc = x - flare_ellipce.center_xy[0]
        yc = y - flare_ellipce.center_xy[1]

        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle

        rad_cc = (xct**2/(flare_ellipce.width/2.)**2) + (yct**2/(flare_ellipce.height/2.)**2)
        distance = np.sqrt(xc**2 + yc**2)

        selection_rule = rad_cc<=1.0
        number_of_targets = len(rad_cc[selection_rule])
        print("number_of_targets:", number_of_targets, " within  ", persentage, " interval")

        data_stars_within_99_percent = 0
        distances_within_99_percent = []

        nearest_star_data = 0
        nearest_star_distance = []
        if number_of_targets > 0:
            indxs = np.argwhere(selection_rule)
            print("Gmag:",  np.array(data_gaia["Gmag"])[indxs])
            print("GAIA ID:",  np.array(data_gaia["Source"])[indxs])
            selcted_distance = list(distance[indxs])
            print(" distance from the ellipce center:",  selcted_distance, " degrees ")
            if persentage == 0.999:
                data_stars_within_99_percent = np.array(data_gaia["Gmag"])[indxs]
                distances_within_99_percent  = (distance[indxs])
        else:
            print("0 stars... within 99.9 persent! take the nearest" )
            indxs = np.argmin(distance)
            nearest_star_data = np.array(data_gaia["Gmag"])[indxs]
            nearest_star_distance = (distance[indxs])
        print("\n")
        return {"data_99_percent":data_stars_within_99_percent, "dist_99_percent":distances_within_99_percent,
                "data_nearest":nearest_star_data, "dist_nearest":nearest_star_distance}






