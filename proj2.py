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
fig.subplots_adjust(left=leftmargin, bottom=bottommargin)  # adjust subplot margins for centering

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


# Step 3: Query & plot photometric redshift catalog sources onto RGB image
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
from astropy.coordinates import SkyCoord, match_coordinates_sky
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
z_phot_vizier = Vizier(columns=['RAJ2000', 'DEJ2000', 'Vmag', 'zb', 'zbmin', 'zbmax'],
           column_filters={"Vmag":"<24"}, row_limit=z_phot_rowlimit)
z_phot_list = z_phot_vizier.get_catalogs("J/AJ/132/926/catalog")[0]  # 0th index is the actual contents of the tablelist
# "zb" = photometric redshift, "RAJ2000" and "DEJ2000" both in degrees (float values)
# "zbmin" and "zbmax" represent lower & upper bounds for zb, respectively

# Assume that all spectroscopic redshift sources have photometric counterparts.
# Hence, we refrain from querying the spectroscopic redshift catalog (commented out below)
'''
##### Spectroscopic redshift catalog #####
# ADS:  https://ui.adsabs.harvard.edu/abs/2017A%26A...608A...2I/abstract
# VizieR:  https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/A%2bA/608/A2/combined
z_spec_rowlimit = -1
# set up filters and limits before querying VizieR.
z_spec_vizier = Vizier(columns=['RAJ2000', 'DEJ2000', 'F775W', 'zMuse'],
           column_filters={"F775W":"<23"}, row_limit=z_spec_rowlimit)
z_spec_list = z_spec_vizier.get_catalogs("J/A+A/608/A2/combined")[0]  # 0th index is the actual contents of the tablelist
# "zMuse" = spectroscopic redshift, "RAJ2000" and "DEJ2000" both in degrees (float values)
'''

##### RA/Dec cross-matching between Photometric & Spectroscopic sources  #####
# https://astroquery.readthedocs.io/en/latest/xmatch/xmatch.html
cross_match_result = XMatch.query(
    cat1 = z_phot_list,
    cat2 = 'vizier:J/A+A/608/A2/combined',  # this is the spectroscopic redshift catalog
    max_distance = 0.1 * u.arcsec,
    colRA1 = 'RAJ2000',
    colDec1 = 'DEJ2000'
)

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

    if z_phot_list['RAJ2000'][i] in cross_match_result['RAJ2000_1']:
        # plot green circles for sources with both phot & spec z
        both_z_circle = patches.Circle((ra_px, dec_px),
                    radius=80, edgecolor='#00FF00', facecolor='none', lw=0.5, alpha=1, label='Both phot & spec z')
        ax.add_patch(both_z_circle)
        
    else:
        # plot magenta circles for sources with both phot & spec z
        phot_z_circle = patches.Circle((ra_px, dec_px),
                    radius=80, edgecolor='#FF00FF', facecolor='none', lw=0.5, alpha=1, label='Phot z only')
        ax.add_patch(phot_z_circle)

    # break the loop if we have too many sources to plot
    if i >= 200:
        break
    
fig.legend(handles=[both_z_circle, phot_z_circle], fontsize=6)  # legend for patch colors
fig.savefig("HUDF.pdf", dpi=400)  # saving only works before showing plot


# Step 4: Create scatterplot of photometric vs. spectroscopic redshifts
fig_scat, ax_scat = plt.subplots(figsize=(6.5, 4))
fig_scat.subplots_adjust(left=0.15, bottom=0.12)
# Create the scatter plot with errorbars
scatterplot = ax_scat.errorbar(cross_match_result['zb'], cross_match_result['zMuse'],
                xerr = [cross_match_result['zb']-cross_match_result['zbmin'], cross_match_result['zbmax']-cross_match_result['zb']],  # asymmetric error bars
                yerr = 0.01*np.abs(cross_match_result['zMuse']),  # placeholder
                fmt='o', markersize=2.5, color='black', alpha=0.5,
                capsize=1.5, ecolor='lightcoral', label="Datapoint"
                )
# Get maximum photometric redshift (max_phot_z)
max_phot_z = int(np.max(cross_match_result['zb']))  # get the max redshift, convert to int
max_range = 2 * max_phot_z + 1
# Overlay a y=x diagonal line
x_function = np.linspace(0, max_phot_z, 100)
linefit = ax_scat.plot(x_function, x_function,
        color='orange', linestyle='--', alpha=0.5,
        label='Line of equality (z_phot = z_spec)')

# Add labels and title
ax_scat.set_xlabel("Photometric redshift (z_phot)")
ax_scat.set_xticks(np.arange(max_range) * 0.5)  # ticks of 0.5 spacing, from 0 to 3.0
ax_scat.set_xlim([0, max_phot_z])  # scake plot according to max phot_z

ax_scat.set_ylabel("Spectropscopic redshift (z_spec)")
ax_scat.set_yticks(np.arange(max_range) * 0.5)  # ticks of 0.5 spacing, from 0 to 3.0
ax_scat.set_ylim([0, max_phot_z])  # scake plot according to max phot_z

ax_scat.set_title("Photometric vs. Spectroscopic Redshifts\nin the Hubble Ultra Deep Field")
ax_scat.set_aspect('equal', adjustable='box')  # force equal scale
handles, labels = ax_scat.get_legend_handles_labels()
ax_scat.legend(handles, labels,  fontsize=6, loc='upper right')
fig_scat.savefig("phot-spec_redshift_plot.pdf", dpi=200)  # saving only works before showing plot

plt.show()