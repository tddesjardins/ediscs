# ediscs
##Introduction
The code in this repository is used in the reduction and analysis of telescope imaging data associated with the ESO Distant Cluster Survey (EDisCS; White et al. 2005). This repository is a work in progress and will be updated quickly over the next several days. In addition, changes will be made to the Python modules themselves (largely simplification and additional commenting for ease of reading) and will result in several updates to the code files. 

##Dependencies
Many of the codes require the use of `PyRAF` and its interface to `IRAF` as well as other standard astronomical packages such as `PyFITS` and `astropy`. A full list of dependencies will be added here in the near future and the depencies of each individual code will be added in the code descriptions section.

##Code descriptions
- `eazy.py` extracts photometry from the FITS tables and puts it into ASCII format for input into EAZY. Can also add additional uncertainty to all filters as well as setting a minimum 10% uncertainty for R-band photometry calibrated off of the synthetic magnitudes computed by G. Rudnick. Finally, this code also plots the results of EAZY by borrowing heavily from gbrammer/threedhst. Note that it is necessary to install `threedhst` for several dependencies.
- `labbe_depth.py` is the empty aperture simulation code. The program places apertures of various sizes on an image and using the method of Labbe et al. (2003) fits a Gaussian profile to the histogram of the sum of the pixel values in the apertures. Multiple aperture sizes are used and a histogram fit for each aperture size. A function is then fit to the widths of the Gaussian fits to the histograms as a function of the aperture size to use in estimating the uncertainty in an arbitrarily sized aperture in the image. The code also allows the user to provide an exposure map to select only regions of the image that have high exposure time. The code returns the coefficients of the fit to the Gaussian widths as a function of aperture size as well as a PDF file of a figure similar to Figure 4 of Labbe et al. (2003).
    - **Dependecies:** `numpy`, `scipy`, `astropy`, `pyfits`, `matplotlib`, `photutils`
- `mosaic_combine.py` stacks multiple MOSAIC 1.1 images from NOAO together. The code uses IRAF tasks to combine the CCDs from calibrated MEF MOSAIC images into a single extension, aligns the images to the USNO B-1.0 astrometric catalog using `WCSTools` (note that the path to WCSTools will need to be updated for each individual installation), and then stacks the images together into a final mosaic of the entire field. The code uses the IRAF tool `mscimatch` to match intensities between images prior to stacking. The first image in the list of input images is assumed to be the reference image and thus the zeropoint for the photometric calibration of the final stack is the same as the zeropoint of the reference image.	

##References
- [Labbe et al. 2003](http://adsabs.harvard.edu/abs/2003AJ....125.1107L), "Ultradeep Near-Infrared ISAAC Observations of the Hubble Deep Field South: Observations, Reduction, Multicolor Catalog, and Photometric Redshifts"

- [White et al. 2005](http://adsabs.harvard.edu/abs/2005A%26A...444..365W), "EDisCS - the ESO distant cluster survey. Sample definition and optical photometry", A&A, 444, 365

##License
The MIT License

Copyright (c) 2016 Tyler D. Desjardins

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
