from astropy.coordinates import SkyCoord, Angle
import numpy as np
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

customSimbad = Simbad()
customSimbad.add_votable_fields('ra')
customSimbad.add_votable_fields('dec')


class StarPositionsFromGAIA:
    def _get_data_from_vizier(self, ra: float=None, dec: float=None, radius_arcsec: float=None):
        coord = SkyCoord(ra, dec, unit='deg')
        vquery = Vizier(columns=['Source', 'RA_ICRS', 'DE_ICRS', '_r',
                                 'Gmag', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE', 'pmRApmDEcor', 'Plx', 'e_Plx',
                                 'PlxpmRAcor', 'PlxpmDEcor'])
        data = vquery.query_region(coord, radius=Angle(radius_arcsec, "arcsec"), catalog=["I/345/gaia2"])
        result = data["I/345/gaia2"].to_pandas()
        return result


    def get_gaia_data(self, tpf_object=None, radius_arcsec=15):
        data = self._get_data_from_vizier(tpf_object.ra, tpf_object.dec, radius_arcsec=radius_arcsec)
        ra_dec_pix = tpf_object.wcs.all_world2pix(np.vstack([data['RA_ICRS'], data['DE_ICRS']]).T, 0)
        ra_dec_pix[:, 0] = ra_dec_pix[:, 0] + tpf_object.column
        ra_dec_pix[:, 1] = ra_dec_pix[:, 1] + tpf_object.row
        return ra_dec_pix, data
