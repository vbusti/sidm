#!/usr/bin/env python
# encoding: UTF8
#

from __future__ import print_function
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage
from astropy.stats import sigma_clipped_stats
from photutils import make_source_mask
from photutils import Background2D, SigmaClip, MedianBackground
from photutils import detect_sources
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.utils import random_cmap
from photutils import detect_threshold
from photutils import source_properties, properties_table
from photutils import source_properties, properties_table
from photutils import EllipticalAperture
from scipy.ndimage import rotate
import statsmodels.api as sm
from scipy.stats import norm
import scipy as sc
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip as ast_sc

with open("my_files.txt", "r") as ins:
    lfiles = []
    for line in ins:
        lfiles.append(line)

wf  = []
w2f = []

for f in lfiles:
    
    hdu = fits.open('/home/vinicius/Documents/sidm/data/i_band/'+str(f[:-1]))[0]
    wcs = WCS(hdu.header)

    data = hdu.data

    #print(hdu.header)

    sigma_clip = SigmaClip(sigma=3., iters=10)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (25, 25), filter_size=(3, 3),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    #print(bkg)

    threshold = bkg.background + (3. * bkg.background_rms)

    sigma = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    segm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel)

    print('labels = ',np.max(segm.labels))

    rand_cmap = random_cmap(segm.max + 1, random_state=12345)
    #norm = ImageNormalize(stretch=SqrtStretch())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.imshow(data, origin='lower', cmap='Greys_r')#, norm=norm)
    ax2.imshow(segm, origin='lower', cmap=rand_cmap)
    plt.savefig('figs/'+str(f)+'fig2.png')


    props = source_properties(data, segm)
    tbl = properties_table(props)
    print(tbl)

    my_min = 100000.


    #props = source_properties(data, segm)
    r = 2.    # approximate isophotal extent
    apertures = []
    for prop in props:
        position = (prop.xcentroid.value, prop.ycentroid.value)
        a = prop.semimajor_axis_sigma.value * r
        b = prop.semiminor_axis_sigma.value * r
        theta = prop.orientation.value
        print(position,theta)
        apertures.append(EllipticalAperture(position, a, b, theta=theta))
        my_dist = np.sqrt((prop.xcentroid.value - 44.)**2+ (prop.ycentroid.value - 44.)**2)
        if(my_dist < my_min):
            my_label = prop.id - 1
            my_min = my_dist

    mytheta = props[my_label].orientation.value

    rand_cmap = random_cmap(segm.max + 1, random_state=12345)
    #norm = ImageNormalize(stretch=SqrtStretch())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.imshow(data, origin='lower', cmap='Greys_r')#, norm=norm)
    ax2.imshow(segm, origin='lower', cmap=rand_cmap)
    for aperture in apertures:
        aperture.plot(color='blue', lw=1.5, alpha=0.5, ax=ax1)
        aperture.plot(color='white', lw=1.5, alpha=1.0, ax=ax2)
    plt.savefig('../figs/'+str(f)+'fig3.png')
    plt.close()

    data3 = props[my_label].make_cutout(data-bkg.background)

    plt.figure()
    data4 = rotate(data3, np.rad2deg(mytheta))
    #norm = ImageNormalize(stretch=SqrtStretch())    
    plt.imshow(data4, origin='lower', cmap='Greys_r')#, norm=norm) 
    plt.savefig('../figs/'+str(f)+'fig4.png')

    a = data4.shape[1]#np.shape(data4[1])
    b = data4.shape[0]#np.shape(data4[0])

    mymu     = np.zeros(a)
    mysigma  = np.zeros(a)
    mypixel  = np.zeros(a)
    mypixerr = np.zeros(a)

    for i in range(a):
    
        g = models.Gaussian1D(amplitude=90.1,mean=np.float(a)/2.,stddev=3.2)
        #arrd = np.linspace(0,126,127)
        datad = np.array(data4[:,i])
        print('len=',len(datad))
        print(datad)
        #datad = datad[datad != 0]
        print(datad)
        arrd = range(len(datad))#np.linspace(0,len(datad)-1,len(datad))
        print(arrd) 
        
        fit = fitting.LevMarLSQFitter()
        fitted_model = fit(g, arrd, datad)
        mymu[i]    = fitted_model.mean.value
        mysigma[i] = np.abs(fitted_model.stddev.value)
        print(mymu[i],mysigma[i]) 
        print(fitted_model)
    
        #arr2 = np.linspace(0,len(datad),len(datad)+1)
        plt.figure()
        #plt.plot(arr2,mymax*((np.sqrt(2.*np.pi)*mysigma[60])**(-1.))*np.exp(-0.5*((arr2-mymu[60])/mysigma[60])**2))
        plt.plot(arrd,fitted_model(arrd),label="No removal")
        plt.scatter(arrd,np.array(data4[:,i]))
        plt.legend()
        plt.savefig('../fig_gauss/'+str(f)+'fig_'+str(i)+'.png')

    arr2 = range(a)

    plt.figure()
    #norm = ImageNormalize(stretch=SqrtStretch())    
    plt.imshow(data4, origin='lower', cmap='Greys_r')#, norm=norm) 
    #plt.errorbar(arr2,mymu,yerr=mysigma,fmt='o',markersize=0.1,lw=0.5)
    plt.plot(arr2,mymu,markersize=0.1,lw=1)
    plt.savefig('../figs/'+str(f)+'fig5.png')


    plt.figure()
    #norm = ImageNormalize(stretch=SqrtStretch())    
    plt.imshow(data4, origin='lower', cmap='Greys_r')#, norm=norm) 
    plt.errorbar(arr2,mymu,yerr=mysigma,fmt='o',markersize=0.1,lw=0.5)
    #plt.plot(arr2,mymu,markersize=0.1,lw=1)
    plt.savefig('../figs/'+str(f)+'fig6.png')

    #mymup = np.zeros(len(mymu))
    mymup = mymu - np.float(b)/2.
    print(mymu)

    x = np.arange(-a/2,a/2,1)

    w = 0.
    w2 = 0.
    for i in range(a/2,len(x)):
        w += 2.*x[i]*mymup[i]/((np.float(a)/2.)**3)

    for i in range(0,len(x)):
        w2 += np.abs(x[i])*mymup[i]/((np.float(a)/2.)**3) 

    print(w,w2)

    wf.append(w)
    w2f.append(w2)


wf  = np.array(wf)
w2f = np.array(w2f)

np.savetxt('../output/w_w2.txt',np.array([wf,w2f]).T)

plt.figure()
plt.hist(wf)
plt.savefig('../figs/histw.png')

plt.figure()
plt.hist(w2f)
plt.savefig('../figs/histw2.png')



















