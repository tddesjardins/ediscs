from astropy.io import fits
from photutils import aperture_photometry
from photutils import CircularAperture
from scipy.optimize import curve_fit

import numpy as np


def labbe_error_function(n, s, a, b):
    """
    PURPOSE
    -------

    NOTES
    -----
    Equation 3 from Labbe et al. (2003), ApJ, 125, 1107.
    """

    error = n * s * (a + (b * n))

    return error


def idl_gauss(x, height, center, sigma):
    """
    PURPOSE
    -------

    NOTES
    -----
    This is the form of the 3-term Gaussian function from IDL.
    """

    z = (x - center) / sigma
    func = (height * np.exp((-1 * (z**2))/2))

    return func


class Labbe(object):

    def __init__(self, image, segmap, expmap=None, aperture_range=[0.5, 3.0], pixel_scale=0.4, sample_size=8):

        # Check inputs.
        if (type(aperture_range) is not list) or (len(aperture_range) != 2):
            raise ValueError('The aperture_range keyword must be a list of [min, max] aperture radii in arcseconds.')

        if sample_size > 8:
            raise ValueError('sample_size = {}, but must be <= 8.'.format(sample_size))

        self.image_name = image
        self.segmap_name = segmap
        self.expmap_name = expmap

        with fits.open(self.image_name) as dhdu:
            with fits.open(self.segmap_name) as shdu:
                self.data_array = dhdu[0].data
                self.segmap_array = shdu[0].data

        self.array_size = np.size(self.data_array)

        # Add in lists for source positions and masking. Convert
        # the aperture size from arcseconds to pixels and get the
        # values of the aperture sizes.
        self.sources_y = []
        self.sources_x = []
        self.sources_mask = None

        aperture_minmax = np.array(aperture_range) / pixel_scale
        self.aperture_sizes = np.logspace(np.log10(aperture_minmax[0]), np.log10(aperture_minmax[1]), sample_size)

        # Set up the photometry table for later.
        self.phot_table = None

    def do_photometry(self, n, buffer=500, exp_frac=0.9):

        # Check inputs.
        if (buffer <= self.aperture_sizes[0]) or (buffer <= self.aperture_sizes[1]):
            raise ValueError('A buffer of {} pixels was requested, but image size '
                             'is (y, x) = ({}, {})!'.format(buffer, self.aperture_sizes[0], self.aperture_sizes[1]))

        if (exp_frac < 0) or (exp_frac > 1):
            raise ValueError('exp_frac = {}, but must be between 0 and 1!'.format(exp_frac))

        # Generate random source positions and mask array.
        self.sources_y += np.random.sample(size=n) * (self.array_size[0] - (2 * buffer)) + buffer
        self.sources_x += np.random.sample(size=n) * (self.array_size[1] - (2 * buffer)) + buffer
        self.sources_mask = np.full(n, False)

        # If an exposure map was provided, open it now and get the data
        # array and maximum value.
        exp_max = None
        if self.expmap_name is not None:
            with fits.open(self.expmap_name) as ehdu:
                expmap_array = ehdu[0].data
            exp_max = np.max(expmap_array)

        # Mask sources. Sources may be masked for one of the
        # following reasons:
        #   1. If an exposure map was provided and aperture is < exp_frac of the maximum
        #   2. If an aperture is too close to a real source
        #   3. If an aperture is too close to another aperture that is already marked as good

        for i, _ in enumerate(self.sources_y):
            y = self.sources_y[i]
            x = self.sources_x[i]

            while not self.sources_mask[i]:

                # Check exposure map if provided.
                if exp_max is not None:
                    if expmap_array[y, x] < (exp_frac * exp_max):
                        self.sources_mask[i] = True
                        break

                # Check if aperture is too close to a real source.
                segphot = self._get_photsum(self.segmap_array, y, x, self.aperture_sizes[-1])
                if segphot > 0:
                    self.sources_mask[i] = True
                    break

        # Do a second pass now and mask all instances where an aperture is
        # too close to another aperture that is already good.

        for i, _ in enumerate(self.sources_y):
            y = self.sources_y[i]
            x = self.sources_x[i]

            while not self.sources_mask[i]:

                distances = ((x - self.sources_x)**2) + ((y - self.sources_y)**2)
                if np.any(distances < self.aperture_sizes[-1]):
                    self.sources_mask[i] = True

        self.phot_table = Table([self.sources_y, self.sources_x], names=('Y', 'X'))
        good = np.where(self.sources_mask == 0)
        for size in self.aperture_sizes:
            aperture_sum = self._get_photsum(self.data_array, self.sources_y[good], self.sources_x[good], size)
            self.phot_table['r{}'.format(size)] = aperture_sum

    def _get_photsum(self, data, y, x, radius):

        positions = []
        for i, _ in enumerate(y):
            positions += [(x[i], y[i])]
        apertures = CircularAperture(positions, r=radius)

        phot_table = aperture_photometry(data, apertures)
        aperture_sum = phot_table['aperture_sum']

        return aperture_sum

