import numpy as np
import scipy.stats as st
import pyfits

#np.random.seed(1234)

nx = 4096/2.
ny = 4096/2.

# Initialise parameters
nstars = 40
seeingx = 5
seeingy = 5

# Draw randomly star positions.
x = (np.random.rand(nstars)*nx).astype(int)
y = (np.random.rand(nstars)*ny).astype(int)

# Draw magnitudes from exponential distribution.
mag = 10 - np.random.exponential(3.0, size=nstars)

# initialize field
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
#fieldintensity = np.zeros((nx, ny), dtype=float)
fieldintensity = np.random.randn(nx, ny)*1e-3

for i in range(nstars):
    print i
    fieldintensity += np.exp(-0.4*mag[i]) * np.exp(-0.5*((X - x[i])**2/seeingx**2 + (Y - y[i])**2/seeingy**2))

# Add bias
fieldintensity += 400

# Write FITS file
outputfile = '/Users/rodrigo/pht/simul_image.fits' 
hdu = pyfits.PrimaryHDU(fieldintensity)
hdu.writeto(outputfile, clobber=True)
