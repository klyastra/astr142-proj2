import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits
import numpy as np
# from astropy.wcs import wcs
# from astropy.table import Table

# Step 1: get the HUDF images in 3 filters.
# The original press release image (https://science.nasa.gov/asset/hubble/hubble-ultra-deep-field-image-reveals-galaxies-galore/)
# states that the ACS/WFC instrument took the HUDF images between Sep 2003 and Jan 2004.
# NASA description says HUDF images come from proposal 9978: 
# https://mast.stsci.edu/search/ui/#/hst/results?proposal_id=9978
# We'll choose the following files:
## J8M802020 - F606W (green --> BLUE), 81.0 min exptime, 2003-09-25 20:09:20 UT
## J8M803020 - F775W (red --> GREEN), 81.0 min exptime, 2003-09-27 00:57:17 UT
## J8M804020 - F814W (IR --> RED), 81.0 min exptime, 2003-09-27 18:10:20 UT
# We'll get the calibrated files in DRC format
# Each DRC-calibrated file has an HDUList containing the science image (0), and uncertainty stuff (1 and 2).
# Read the HUDF paper for more info https://iopscience.iop.org/article/10.1086/507302/pdf

# Step 2: Plot each image with a unique colormap tint and then additively overlay together to form a color composite.
# BLUE = J8M802020
blue_data = fits.getdata('j8m802020_drc.fits')
green_data = fits.getdata('j8m803020_drc.fits')
red_data = fits.getdata('j8m804020_drc.fits')

# setup the figure
fig, ax = plt.subplots(figsize=(6.5, 4))
leftmargin = 0.08
bottommargin = 0.16
plt.subplots_adjust(left=leftmargin, bottom=bottommargin)  # adjust subplot margins for centering


# RGB color is defined by a 3-list. Create an RGB image by assigning R, G, B:
rgb_data = np.stack([red_data, green_data, blue_data], axis=-1)

# Define a function that allows us the modify levels of each channel in an RGB image
# https://stackoverflow.com/questions/42008932/how-to-change-vmin-and-vmax-of-each-color-with-matplotlib-imshow
def channelnorm(im, channel, vmin, vmax):
    c = (im[:,:,channel]-vmin) / (vmax-vmin)
    c[c<0.] = 0
    c[c>1.] = 1
    im[:,:,channel] = c
    return im

# Stretched red channel
rgb_data = channelnorm(rgb_data, 0, -0.01, 0.043)
# Stretched green channel
rgb_data = channelnorm(rgb_data, 1, -0.01, 0.052)
# Stretched blue channel
rgb_data = channelnorm(rgb_data, 2, -0.01, 0.05)

# Display RGB image after modifying channels
rgb_display = ax.imshow(rgb_data, origin='lower')

#vmin = 0  # set minimum brightness
#vmax = 0.1  # set maximum brightness
#blue = ax.imshow(blue_data, cmap='twilight_shifted', origin='lower', vmin=vmin, vmax=vmax)

# Create a separate Axes for the colorbar, defining its position and size
# The arguments are [left, bottom, width, height] in figure coordinates (0 to 1)
#fig.colorbar(blue, label='Pixel Flux (datanumber)')

# Then set title and labels
ax.set_title('Hubble Ultra Deep Field')
ax.set_xlabel('X position (pixels)') # Label the x-axis
ax.set_ylabel('Y position (pixels)') # Label the y-axis

plt.show()