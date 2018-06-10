"""
These functions are small tools that were originally in the ediscs_catalog.py program,
but have been moved here to simplify that code.
"""

from astropy import units as u
from astropy import coordinates as coords
from sfdmap import ebv

import numpy as np
import random
import os


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


def labbe_sigfunc(n, s, a, b):
    """
    PURPOSE
    -------
    Function from equation 3 from Labbe et al. (2003), ApJ, 125, 1107.
    """

    value = n * s * (a + (b * n))

    return value


def get_ebv(ra, dec, frame='fk5'):
    """
    PURPOSE
    -------
    Determine the E(B-V) value from the Schlegel dust maps. Must set SFD_DIR environment
    variable pointing to the Schlegel map FITS files directory prior to running.

    INPUTS
    ------
    ra (str or astropy.units object):
        The right ascension of the source. This can be specified as either a string
        or an astropy.units object. If it is a string, it must specify the units,
        e.g., '10h18m48.6s'.

    dec (str or astropy.units object):
        The declination of the source. This can be specified as either a string or an
        astropy.units object. If it is a string, it must specify the units, e.g., '-12d11m53s'.

    frame (str):
        Name of the coordinates frame. Default is 'fk5'.

    RETURNS
    -------
    ebv (float):
        E(B-V) reddening at the source coordinates.
    """

    if 'SFD_DIR' not in os.environ:
        raise OSError('Must set SFD_DIR environment variable!')

    galcoord = coords.SkyCoord(ra=ra, dec=dec, frame=frame).galactic
    ebv_value = ebv(galcoord) / 0.86

    return ebv_value


def matchxy(x1, y1, x2, y2, tol=0.1):
    """
    PURPOSE
    -------
    For a given set of (x,y) coordinates, find the index of the best match from
    a list of (x,y) coordinates.

    INPUTS
    ------
    x1 (float):
        X position of interest.

    y1 (float):
        Y position of interest.

    x2 (numpy.array):
        Numpy array of X positions (floats) that are being matched against.

    y2 (numpy.array):
        Numpy array of Y positions (float) that are being matched against.

    tol (float):
        Matching tolerance.

    RETURNS
    ------
    best_match (int):
        The index in the arrays being matched against containing the best match.
        If no match is <= the tol value, then None is returned.
    """

    distance = np.sqrt((x1 - x2)**2 + (y1-y2)**2)
    if np.min(distance) <= tol:
        best_match = np.where(distance == np.min(distance))[0][0]
    else:
        best_match = None

    return best_match


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


def random_sample(catalog, filters, output1='randomFirst.fits', output2='randomSecond.fits', class_star=0.3, q=4):
    """
    PURPOSE: Generate a random sample of galaxies from the photometric catalog of a single cluster.
    \tAlso generate a second FITS file of the other galaxies not in the random sample.'

    INPUTS:
    \tcatalog   - FITS photometric catalog
    \tfilters   - string list of the filter names (e.g., BVRIzK)
    \toutput1   - Name of random sample FITS catalog
    \toutput2   - Name of FITS catalog with galaxies NOT in the random sample
    \tclassStar - SExtractor class_star value to perform cut on (selects objects with class_star < this value)
    \tQ         - LDP quality flag for selecting sources

    RETURNS: None.
    """

    select = ('class_StarR < {} & Q5_7 == {} & '.format(class_star, q) +
              ' & '.join(['sexflag{} == 0'.format(x) for x in filters]))

    data = Table.read(catalog).to_pandas()
    trim_data = data.query(select)

    idx = xrange(len(trim_data))
    sample = random.sample(idx, int(np.ceil(len(trim_data) / 2.)))
    inverse = []
    for x in idx:
        if x not in sample:
            inverse.append(x)
    first = trim_data.iloc[sample]
    second = trim_data.iloc[inverse]

    fits_cols = list(trim_data.columns.values)
    first_dict = {col_names[x]: first[fits_cols[x]].values for x in range(len(fits_cols))}
    second_dict = {col_names[x]: second[fits_cols[x]].values for x in range(len(fits_cols))}
    gals_dict = {col_names[x]: trim_data[fits_cols[x]].values for x in range(len(fits_cols))}

    Table(first_dict, names=col_names, dtype=Fdtype).write(output1, format='fits', overwrite=True)
    Table(second_dict, names=col_names, dtype=Fdtype).write(output2, format='fits', overwrite=True)
    Table(gals_dict, names=col_names, dtype=Fdtype).write('galaxies.fits', format='fits', overwrite=True)
