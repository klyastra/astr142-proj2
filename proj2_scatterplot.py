import matplotlib.pyplot as plt
import numpy as np
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
## print(z_phot_list['RAJ2000'], z_phot_list['DEJ2000'])

# plot circle patches for each row from the VizieR table
for i in range(len(z_phot_list['Vmag'])):
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

    # plot the circle patch at our converted locations (colored magenta for z_phot)
    z_phot_circle = patches.Circle((ra_px, dec_px),
                    radius=80, edgecolor='#FF00FF', facecolor='none', lw=0.5, alpha=0.5, label="z_phot only")
    ax.add_patch(z_phot_circle)

# Step 4: Query & plot spectroscopic redshift catalog sources onto RGB image
##### Spectroscopic redshift catalog #####
# ADS:  https://ui.adsabs.harvard.edu/abs/2017A%26A...608A...2I/abstract
# VizieR:  https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/A%2bA/608/A2/combined
z_spec_rowlimit = -1
# set up filters and limits before querying VizieR.
z_spec_vizier = Vizier(columns=['RAJ2000', 'DEJ2000', 'F775W', 'zMuse'],
           column_filters={"F775W":"<23"}, row_limit=z_spec_rowlimit)
z_spec_list = z_spec_vizier.get_catalogs("J/A+A/608/A2/combined")[0]  # 0th index is the actual contents of the tablelist
# "zMuse" = spectroscopic redshift, "RAJ2000" and "DEJ2000" both in degrees (float values)
print(z_spec_list['RAJ2000'], z_spec_list['DEJ2000'])

# plot circle patches for each row from the VizieR table
for i in range(len(z_spec_list['RAJ2000'])):
    ra_float = z_spec_list['RAJ2000'][i]
    dec_float = z_spec_list['DEJ2000'][i]

    # Convert float to RA/Dec with units
    skycoord_object = SkyCoord(ra=ra_float * u.degree, dec=dec_float * u.degree, frame='icrs')
    ## ra, dec = skycoord_object.ra, skycoord_object.dec
    # Plot the circle patch.
    # By default, the circle patch's (x,y) coordinate argument is in units of pixels.
    # Recall that the image size in pixels is 4217 x 4243 (in x,y orientation)
    # We have RA/Dec coords from VizieR. Convert RA/Dec into pixels according to our WCS object named "w".
    ra_px, dec_px = w.world_to_pixel(skycoord_object)

    # plot the circle patch at our converted locations (colored yellow for z_phot)
    z_spec_circle = patches.Circle((ra_px, dec_px),
                    radius=80, edgecolor='#00FF00', facecolor='none', lw=0.5, alpha=0.5, label="z_spec only")
    ax.add_patch(z_spec_circle)