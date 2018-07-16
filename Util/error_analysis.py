from astropy.io import fits
from astropy.stats import biweight_location
from astropy.stats import biweight_midvariance
from astropy.table import Table
from photutils import aperture_photometry
from photutils import CircularAperture
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np


def edges_to_centers(edges):

    centers = np.empty(len(edges) - 1)
    for i, x in enumerate(centers):
        centers[i] = ((edges[i + 1] - edges[i]) / 2) + edges[i]

    return centers


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

        self.array_size = np.shape(self.data_array)

        # Add in lists for source positions and masking. Convert
        # the aperture size from arcseconds to pixels and get the
        # values of the aperture sizes.
        self.sources_y = None
        self.sources_x = None
        self.sources_mask = None
        self.good_sources = 0

        # Make logarithmically spaced apertures. Round the aperture
        # size to the 5th decimal place to make it more compact for
        # the table column names.
        aperture_minmax = np.array(aperture_range) / pixel_scale
        self.aperture_sizes = np.logspace(np.log10(aperture_minmax[0]), np.log10(aperture_minmax[1]), sample_size)
        self.aperture_sizes = np.around(self.aperture_sizes, 5)

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
        self.sources_y = np.random.sample(size=n) * (self.array_size[0] - (2 * buffer)) + buffer
        self.sources_x = np.random.sample(size=n) * (self.array_size[1] - (2 * buffer)) + buffer
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

        print('Checking if fake apertures are (1) in low exposure areas [optional] or (2) too close to a real source.')

        # Check exposure time fraction if exposure map is provided.
        for i, _ in enumerate(self.sources_y):
            y = self.sources_y[i]
            x = self.sources_x[i]

            if self.expmap_name is not None:
                while not self.sources_mask[i]:
                    if expmap_array[y, x] < (exp_frac * exp_max):
                        self.sources_mask[i] = True
                        break

        # Check if aperture is too close to a real source.
        segphot = self._get_photsum(self.segmap_array, self.sources_y, self.sources_x, self.aperture_sizes[-1])
        bad_segmap_results = np.where(segphot > 0)
        self.sources_mask[bad_segmap_results] = True

        temp_good = np.where(self.sources_mask == 0)
        print('\tFound {} out of {} good apertures.'.format(np.size(temp_good), n))
        print('Checking if fake apertures are too close to each other.')

        # Do a second pass now and mask all instances where an aperture is
        # too close to another aperture that is already good.

        for i, _ in enumerate(self.sources_y):
            y = self.sources_y[i]
            x = self.sources_x[i]

            if not self.sources_mask[i]:

                distances = ((x - self.sources_x)**2) + ((y - self.sources_y)**2)
                if np.any((distances < self.aperture_sizes[-1]) & (distances > 0)):
                    self.sources_mask[i] = True

        good = np.where(self.sources_mask == 0)
        self.good_sources = np.size(good)
        self.phot_table = Table([self.sources_y[good], self.sources_x[good]], names=('Y', 'X'))

        print('\tFound {} out of {} good apertures. Using these results.'.format(np.size(good), n))
        print('Doing aperture photometry...')

        for i, size in enumerate(self.aperture_sizes):
            print('\tWorking on r = {} pixels (#{} out of {})'.format(size, i+1, len(self.aperture_sizes)))
            aperture_sum = self._get_photsum(self.data_array, self.sources_y[good], self.sources_x[good], size)
            self.phot_table['r{}'.format(size)] = aperture_sum

        print('Finished!')

    def fit_sigma_function(self, minmax_value=500, outroot='labbe', label=None):

        fig = plt.figure(figsize=(11, 8))
        ax1 = fig.add_subplot(111)
        colors = cmx.jet(np.linspace(0, 1, np.size(self.aperture_sizes)))

        # Make a histogram for each aperture radius and then
        # fit it with a Gaussian to get the value of sigma.
        column_names = self.phot_table.colnames
        gauss_sigma = []
        i = 0
        for column in column_names:
            if 'r' in column:
                good_phot = np.where(np.abs(self.phot_table[column] < minmax_value))
                fitting_data = self.phot_table[column][good_phot]

                hist, bin_edges = np.histogram(fitting_data, bins='sqrt')
                bin_centers = edges_to_centers(bin_edges)

                # Set up initial guesses for the fit.
                init_height = np.max(hist)
                init_center = biweight_location(fitting_data)
                init_sigma = np.sqrt(biweight_midvariance(fitting_data))
                init_guesses = [init_height, init_center, init_sigma]

                # Fit the histogram.
                (gfit, gcovariance) = curve_fit(idl_gauss, bin_centers, hist, p0=init_guesses, maxfev=10000000)
                if gfit[2] < 0:
                    gfit[2] = np.abs(gfit[2])
                gauss_sigma += [gfit[2]]

                xarr = np.linspace(-minmax_value, minmax_value, 20000)
                yarr = idl_gauss(xarr, gfit[0], gfit[1], gfit[2])
                ax1.plot(bin_centers, hist, color=colors[i], drawstyle='steps-mid')
                ax1.plot(xarr, yarr, color=colors[i], linestyle='--')

                i += 1

        # Add some information to the figure.
        if label is not None:
            ax1.set_title('{} Gaussian Fits'.format(label))
        ax1.set_xlabel('Aperture Sum')
        ax1.set_ylabel('N')
        ax1.set_xlim(np.min(bin_centers), np.max(bin_centers))

        fig.savefig('{}_gaussian_hist.png'.format(outroot))
        plt.close(fig)

        print('Saved Gaussian fits to histograms --> {}_gaussian_hist.png'.format(outroot))

        # The "N" in Labbe et al. equation 3 is really the circularized
        # radius.
        labbe_n = np.sqrt(np.pi * (self.aperture_sizes**2))
        (sfit, scovariance) = curve_fit(labbe_error_function, labbe_n, gauss_sigma)

        # Plot the result.
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        splot_max = np.ceil(np.max(labbe_n)) + 2
        xarr = np.linspace(0, splot_max, 1000)
        yarr = labbe_error_function(xarr, sfit[0], sfit[1], sfit[2])
        ax1.plot(labbe_n, gauss_sigma, 'ro')
        ax1.plot(xarr, yarr, 'r--')

        # Add some information to the figure.
        if label is not None:
            ax1.set_title('{} Labbe Error Function Fit'.format(label))
        ax1.set_xlabel('Circularized radius / pixels')
        ax1.set_ylabel('Sigma')

        fig.savefig('{}_errorfunc_fit.png'.format(outroot))
        plt.close(fig)

        print('Saved Labbe error function fit --> {}_errorfunc_fit.png'.format(outroot))

        return sfit

    def _get_photsum(self, data, y, x, radius):

        positions = []
        for i, _ in enumerate(y):
            positions += [(x[i], y[i])]
        apertures = CircularAperture(positions, r=radius)

        phot_table = aperture_photometry(data, apertures)
        aperture_sum = phot_table['aperture_sum']

        return aperture_sum

