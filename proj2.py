import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

### setup the figure
fig = plt.figure(figsize=(8, 5))
fig.tight_layout()
gridspec = gridspec.GridSpec(3, 5, figure=fig) # big plot takes up 3 rows & 3 columns, + 2 columns for zoom plots

### create the big plot with the RGB image
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

# Display RGB image in the big plot after modifying channels. This big plot spans multiple grid cells.
ax_large = fig.add_subplot(gridspec[0:3, 0:3], projection=w) # Spans 3 rows and columns (from index 0 to 2)
ax_large.imshow(rgb_data, origin='lower')

ax_large.set_title('Hubble Ultra Deep Field')
ax_large.set_xlabel('Right Ascension (hms)') # Label the x-axis
ax_large.set_ylabel('Declination (dms)', labelpad=-1.) # Label the y-axis; reduce padding

# Create the six smaller zoom-in plots
# We'll place them in the remaining grid cells and potentially create new ones within those
zoom_regions = [
    (0, 3),  # Top-left corner
    (0, 4),  # Top-right corner
    (1, 3),  # Middle-left corner
    (1, 4),  # Middle-right corner 
    (2, 3),  # Bottom-left corner
    (2, 4)   # Bottom-right corner
]

# Define zoom-in limits for each small plot
zoomin_size = 150  # panel width in pixels
bl_corner_x = [3130, 3555, 1696, 596, 1420, 2000]  # x positions of each panel's bottom left corner
bl_corner_y = [1000, 1650, 3506, 2600, 3500, 1091]  # y positions of each panel's bottom left corner

# list of unique border colors for each zoom-in subplot
from matplotlib.patches import Rectangle
colorlist = ['red', 'orange', 'yellow', 'lime', 'cyan', 'magenta']
letterlist = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
for i, (row_index, col_index) in enumerate(zoom_regions):
    if i < 6: # Ensure we only create 6 zoom-in plots
        ax_zoom = fig.add_subplot(gridspec[row_index, col_index], projection=w)
        ax_zoom.imshow(rgb_data, origin='lower')
        ax_zoom.set_xlim(bl_corner_x[i], bl_corner_x[i]+zoomin_size)
        ax_zoom.set_ylim(bl_corner_y[i], bl_corner_y[i]+zoomin_size)
        ax_zoom.axis('off') # Hide ticks

        # Apply rectangle border for each zoom-in subplot
        border = Rectangle((bl_corner_x[i], bl_corner_y[i]), zoomin_size, zoomin_size,
                        fill=False, edgecolor=colorlist[i], linewidth=5)
        ax_zoom.add_patch(border)

        # Add text at the upper right corner of each zoom-in subplot.
        ax_zoom.text(bl_corner_x[i]+0.9*zoomin_size, bl_corner_y[i]+0.9*zoomin_size,
                letterlist[i], fontsize=12, color=colorlist[i], ha='right', va='top')

        # Plot squares representing areas of zoom on main big figure
        zoom_loc = Rectangle((bl_corner_x[i], bl_corner_y[i]), zoomin_size, zoomin_size,
                        fill=False, edgecolor=colorlist[i], linewidth=0.5)
        ax_large.add_patch(zoom_loc)

plt.subplots_adjust(left=0.16, bottom=0.17, hspace=0.03)  # reduce vertical space
fig.savefig("HUDF_multiplot.pdf", dpi=300)  # saving only works before showing plot
plt.show()