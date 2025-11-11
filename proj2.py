import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits
from astropy.wcs import WCS
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
## We assign colors according to wavelength
# We'll get the calibrated files in DRC format
# Each DRC-calibrated file has an HDUList containing the science image (0), and uncertainty stuff (1 and 2).
# Read the HUDF paper for more info https://iopscience.iop.org/article/10.1086/507302/pdf


# Step 2: Plot each image with a unique colormap tint and then additively overlay together to form a color composite.
# BLUE = J8M802020
blue_data = fits.getdata('j8m802020_drc.fits')
green_data = fits.getdata('j8m803020_drc.fits')
red_data = fits.getdata('j8m804020_drc.fits')

# Obtain the WCS from any one of the FITS images. They all have the same sky location & WCS, so it doesn't matter.
hdulist = fits.open('j8m803020_drc.fits')
header = hdulist[1].header  # Access the header of the science HDU (index 1)
w = WCS(header)  # obtain the WCS object
hdulist.close()  # close the FITS file to save memory
##print(w)

# setup the figure
fig, ax = plt.subplots(figsize=(6.5, 4), subplot_kw=dict(projection=w))  # projection keyword for WCS
leftmargin = 0.13
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

# Then set title and labels
ax.set_title('Hubble Ultra Deep Field')
ax.set_xlabel('Right Ascension (hms)') # Label the x-axis
ax.set_ylabel('Declination (dms)', labelpad=-1.) # Label the y-axis; reduce padding


# Step 3: Retrieve source positions and photometric+spectroscopic redshifts from VizieR catalog
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.patches as patches
from astropy.visualization import quantity_support
quantity_support()  # this enables us to plot astropy Quantity values with units

##### Photometric redshift catalog #####
# ADS:  https://ui.adsabs.harvard.edu/abs/2006AJ....132..926C/abstract
# VizieR:  https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/AJ/132/926/catalog
z_phot_rowlimit = -1
# set up filters and limits before querying VizieR.
# Here we'll only get bright objects so we can see them in our plot.
# PLus, bright objects are more likely to have spectroscopic redshift measurements.
z_phot_vizier = Vizier(columns=['RAJ2000', 'DEJ2000', 'Vmag', 'zb'],
           column_filters={"Vmag":"<23"}, row_limit=z_phot_rowlimit)
z_phot_list = z_phot_vizier.get_catalogs("J/AJ/132/926/catalog")[0]  # 0th index is the actual contents of the tablelist
# "zb" = photometric redshift, "RAJ2000" and "DEJ2000" both in degrees (float values)
print(z_phot_list['RAJ2000'], z_phot_list['DEJ2000'])

# plot circle patches for each row from the VizieR table
for i in range(len(z_phot_list['RAJ2000'])):
    ra_float = z_phot_list['RAJ2000'][i]
    dec_float = z_phot_list['DEJ2000'][i]

    # Convert float to RA/Dec with units
    skycoord_object = SkyCoord(ra=ra_float * u.degree, dec=dec_float * u.degree, frame='icrs')
    ## ra, dec = skycoord_object.ra, skycoord_object.dec
    # Plot the circle patch.
    # By default, the circle patch's (x,y) coordinate argument is in units of pixels.
    # Recall that the image size in pixels is 4217 x 4243 (in x,y orientation)
    # We have RA/Dec coords from VizieR. Convert RA/Dec into pixels according to our WCS object named "w".
    ra_px, dec_px = w.world_to_pixel(skycoord_object)

    # plot the circle patch at our converted locations (colored yellow for z_phot)
    z_phot_circle = patches.Circle((ra_px, dec_px),
                    radius=80, edgecolor='#FFFF00', facecolor='none', lw=0.5, label="z_phot only")
    ax.add_patch(z_phot_circle)

##### Spectroscopic redshift catalog #####


# Step 4: Plot phot/spec redshift catalog sources onto RGB image


plt.legend(handles=[z_phot_circle], fontsize=6)  # legend for patch colors
plt.savefig("HUDF.pdf", dpi=400)  # saving only works before showing plot
plt.show()