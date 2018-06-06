"""
These functions are small tools that were originally in the ediscs_catalog.py program,
but have been moved here to simplify that code.
"""

from astropy import units as u
from astroquery.irsa_dust import IrsaDust
from astropy import coordinates as coords

import numpy as np


def get_zp(flux, mag):
    """
    PURPOSE
    -------
    Backout the zeropoint for a source knowing its flux in detector units and its physical magnitude.

    INPUTS
    ------
    flux (float):
        flux in detector units (counts or counts/second)
    mag (float):
        magnitude in physical units (note that if this is in AB mag, the result includes the AB
        conversion factor)

    RETURNS
    -------
    The zeropoint to convert between detector flux and physical magnitude
    """

    zp = 2.5 * np.log10(flux) + mag

    return zp


def fixed_mag_color(x, zp, type):
    """
    PURPOSE
    -------
    This function is designed to be used for fitting zeropoints by comparing the WFI
    and FORS observations of stars using a constant color term, but allowing the
    zeropoint to vary.

    INPUTS
    ------
    x (float):

    zp (float):

    type (str):
    """

    constants = const.Constants()
    magnitude = zp + (constants.med_cterms(type) * x)

    return magnitude


def labbe_sigfunc(N, s, a, b):
    """
    PURPOSE
    -------
    Function from equation 3 from Labbe et al. (2003), ApJ, 125, 1107.
    """

    value = N * s * (a + (b * N))

    return value


def get_ebv(ra, dec, radius=2*u.deg, frame='fk5'):
    """
    PURPOSE
    -------
    Determine the E(B-V) value from the Schlegel dust maps averaged over some radius.

    INPUTS
    ------
    ra (str or astropy.units object):
        The right ascension of the source. This can be specified as either a string
        or an astropy.units object. If it is a string, it must specify the units,
        e.g., '10h18m48.6s'.

    dec (str or astropy.units object):
        The declination of the source. This can be specified as either a string or an
        astropy.units object. If it is a string, it must specify the units, e.g., '-12d11m53s'.

    radius (astropy.units object):
        The radius over which the E(B-V) is averaged. The input value is an astropy.units
        object (e.g., 2*u.degree). The acceptable range is 2 - 37.5 degrees, and the default
        is 2 degrees.

    frame (str):
        Name of the coordinates frame. Default is 'fk5'.

    RETURNS
    -------
    ebv (float):
        E(B-V) reddening at the source coordinates averaged over the radius.
    """

    galcoord = coords.SkyCoord(ra=ra, dec=dec, frame=frame)
    extinct_tab = IrsaDust.get_query_table(galcoord, section='ebv', radius=radius)
    ebv = extinct_tab['ext SFD mean'][0]

    return ebv


class MagTools(object, flux=0.0, mag=0.0, zp=0.0, ab_conv=0.0):
    """
    An object to contain functions related to transforming between fluxes
    and magnitudes.
    """

    def __init__(self):

        self.flux = flux
        self.mag = mag
        self.zp = zp
        self.ab_conv = ab_conv

    def flux_to_mag(self):
        """
        PURPOSE
        -------
        Calculate a magnitude for a given flux, zeropoint, and AB conversion
        factor.

        RETURNS
        -------
        magnitude (float):
            The magnitude corresponding to the input flux, zeropoint, and AB
            conversion factor.
        """

        magnitude = -2.5 * np.log10(self.flux) + self.zp + self.ab_conv

        return magnitude

    def mag_to_flux(self, ujy=True):
        """
        PURPOSE
        -------
        Calculate the flux of a source given a calibrated physical flux.

        INPUTS
        ------
        ujy (bool):
            If true, the output will be in units of uJy. Otherwise, the flux
            will have the units of the flux reference used to calculate
            the input magnitude.

        RETURNS
        -------
        flux (float):
            The flux that corresponds to the input magnitude.
        """

        flux = 10**(-0.4 * self.mag)
        if ujy:
            flux *= (3631.0 * 1e6)

        return flux

    def physical_flux(self):
        """
        PURPOSE
        -------
        Calculate the calibrated physical flux given an instrumental flux,
        a zeropoint, and an AB magnitude conversion factor.

        RETURNS
        -------
        calib_flux (float):
            A calibrated physical flux in units of uJy.
        """

        self.mag = self.flux_to_mag()
        calib_flux = self.mag_to_flux(ujy=True)

        return calib_flux
