#!/usr/bin/env python
# encoding: UTF8
#

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage
from astropy.stats import sigma_clipped_stats
from photutils.utils import random_cmap
from photutils import detect_threshold, source_properties, properties_table, EllipticalAperture
from photutils import Background2D, SigmaClip, MedianBackground, make_source_mask, detect_sources
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.visualization import SqrtStretch
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip as ast_sc
from scipy.ndimage import rotate
from scipy.stats import norm

def prepare_data(file,data_dir,folder):
    """
    prepare_data picks a file with the image of the galaxy, detect the central object, rotate it to the major axis, and returns 
    the data and errors ready to fit a warp curve, along with the maximum distance from the center 

    """
    hdu = fits.open(data_dir+'/'+str(file[:-1]))[0]
    wcs = WCS(hdu.header)

    data = hdu.data

    weight = fits.open(data_dir+'/'+str(file[:-1]))[1].data

    sigma_clip = SigmaClip(sigma=3., iters=10)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (25, 25), filter_size=(3, 3),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)


    threshold = bkg.background + (3. * bkg.background_rms)

    sigma = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    segm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel)


    rand_cmap = random_cmap(segm.max + 1, random_state=12345)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.imshow(data, origin='lower', cmap='Greys_r')
    ax2.imshow(segm, origin='lower', cmap=rand_cmap)
    plt.savefig('../figs/'+str(folder)+'/'+str(file[:-5])+'fig2.png')


    props = source_properties(data, segm)
    tbl = properties_table(props)

    my_min = 100000.

    r = 3.    # approximate isophotal extent
    apertures = []
    for prop in props:
        position = (prop.xcentroid.value, prop.ycentroid.value)
        #print(position) 
        a = prop.semimajor_axis_sigma.value * r
        b = prop.semiminor_axis_sigma.value * r
        theta = prop.orientation.value
        apertures.append(EllipticalAperture(position, a, b, theta=theta))
        my_dist = np.sqrt((prop.xcentroid.value - 44.)**2+ (prop.ycentroid.value - 44.)**2)
        if(my_dist < my_min):
            my_label = prop.id - 1
            my_min = my_dist
        


    mytheta  = props[my_label].orientation.value
    mysize   = np.int(np.round(r*props[my_label].semimajor_axis_sigma.value))
    my_x     = props[my_label].xcentroid.value
    my_y     = props[my_label].ycentroid.value
   
    mask_obj = np.ones(data.shape,dtype='bool')
    mask_obj[(segm.data != 0)*(segm.data != props[my_label].id)] = 0

    weigth = weight[mask_obj]  


    rand_cmap = random_cmap(segm.max + 1, random_state=12345)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.imshow(data, origin='lower', cmap='Greys_r')
    ax2.imshow(segm, origin='lower', cmap=rand_cmap)
    for aperture in apertures:
        aperture.plot(color='blue', lw=1.5, alpha=0.5, ax=ax1)
        aperture.plot(color='white', lw=1.5, alpha=1.0, ax=ax2)
    plt.savefig('../figs/'+str(folder)+'/'+str(file[:-5])+'fig3.png')
    plt.close()

    data_rot = rotate(data, np.rad2deg(mytheta))
    data_rot = data_rot[data_rot.shape[0]/2 - 30:data_rot.shape[0]/2 + 30,data_rot.shape[1]/2 - 30:data_rot.shape[1]/2 + 30]   


    w_rot  = rotate(weight, np.rad2deg(mytheta))
    w      = w_rot[w_rot.shape[0]/2 - 30:w_rot.shape[0]/2 + 30,w_rot.shape[1]/2 - 30:w_rot.shape[1]/2 + 30] 

    plt.figure()    
    plt.imshow(data_rot, origin='lower', cmap='Greys_r') 
    plt.savefig('../figs/'+str(folder)+'/'+str(file[:-5])+'fig4.png')

    np.savetxt('../output/'+str(folder)+'/'+str(file[:-5])+'_size_xcent_ycent.txt',np.array([mysize,my_x,my_y]))

    return data_rot, w, mysize, my_x, my_y 



