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
from scipy.optimize import curve_fit
import random

def my_chisq_min(x,y,amp,mean,sigma):
    return np.sum( (y - amp*np.exp(- ((x-mean)**2/(2.*sigma**2))) )**2)/(np.float(len(y))-3.)
    
def my_gaussian(x,amp,mean,sigma):
    return amp*np.exp(-(x-mean)**2/(2.*sigma**2)) 

def my_bootstrap(x,y,sigma):
    acc=0
    mean_values = np.zeros(1000)
    while(acc < 1000):
        try:
            x_b    = np.zeros(len(x))
            y_b    = np.zeros(len(x))
            err_yb = np.zeros(len(x)) 

            for i in range(len(x)):
                aux=random.randint(0,len(x)-1)
                x_b[i]    = x[aux] 
                y_b[i]    = y[aux]
                err_yb[i] = sigma[aux] 

            popt, pcov = curve_fit(my_gaussian, x_b, y_b,p0=[40,30,3],sigma=err_yb) 
        except:
            j=0# do nothing  
        else:        
            mean_values[acc]    = popt[1]
            acc += 1 
    return np.std(mean_values)
            

    

with open("my_files.txt", "r") as ins:
    lfiles = []
    for line in ins:
        lfiles.append(line)

folder = 'err_w_I_test'

wf  = []
w2f = []

vec_s  = []
vec_os = []

for f in lfiles:
    
    hdu = fits.open('/home/vinicius/Documents/sidm/data/i_band/'+str(f[:-1]))[0]
    wcs = WCS(hdu.header)

    data = hdu.data

    weight = fits.open('/home/vinicius/Documents/sidm/data/i_band/'+str(f[:-1]))[1].data

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
    plt.savefig('../figs/'+str(folder)+'/'+str(f[:-5])+'fig2.png')


    props = source_properties(data, segm)
    tbl = properties_table(props)

    my_min = 100000.

    r = 3.    # approximate isophotal extent
    apertures = []
    for prop in props:
        position = (prop.xcentroid.value, prop.ycentroid.value)
        a = prop.semimajor_axis_sigma.value * r
        b = prop.semiminor_axis_sigma.value * r
        theta = prop.orientation.value
        apertures.append(EllipticalAperture(position, a, b, theta=theta))
        my_dist = np.sqrt((prop.xcentroid.value - 44.)**2+ (prop.ycentroid.value - 44.)**2)
        if(my_dist < my_min):
            my_label = prop.id - 1
            my_min = my_dist

    mytheta  = props[my_label].orientation.value
    mysize   = np.int(np.round(r*props[my_label].semimajor_axis_sigma.value*np.cos(mytheta)))
    vec_s.append(mysize)
    vec_os.append(a)
    print('mysize = %d and size = %d'% (mysize,a)) 
    
    mask_obj = np.ones(data.shape,dtype='bool')
    mask_obj[(segm.data != 0)*(segm.data != my_label)] = 0

    weigth = weight[mask_obj]  


    rand_cmap = random_cmap(segm.max + 1, random_state=12345)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.imshow(data, origin='lower', cmap='Greys_r')
    ax2.imshow(segm, origin='lower', cmap=rand_cmap)
    for aperture in apertures:
        aperture.plot(color='blue', lw=1.5, alpha=0.5, ax=ax1)
        aperture.plot(color='white', lw=1.5, alpha=1.0, ax=ax2)
    plt.savefig('../figs/'+str(folder)+'/'+str(f[:-5])+'fig3.png')
    plt.close()

    data4 = rotate(data, np.rad2deg(mytheta))
    data4 = data4[data4.shape[0]/2 - 30:data4.shape[0]/2 + 30,data4.shape[1]/2 - 30:data4.shape[1]/2 + 30]

    w_rot  = rotate(weight, np.rad2deg(mytheta))
    w      = w_rot[w_rot.shape[0]/2 - 30:w_rot.shape[0]/2 + 30,w_rot.shape[1]/2 - 30:w_rot.shape[1]/2 + 30] 

    plt.figure()    
    plt.imshow(data4, origin='lower', cmap='Greys_r') 
    plt.savefig('../figs/'+str(folder)+'/'+str(f[:-5])+'fig4.png')


    a = data4.shape[1]
    b = data4.shape[0]

    mymu     = np.zeros(a)
    mysigma  = np.zeros(a)
    myamp    = np.zeros(a)
    mymask   = np.ones(a,dtype='bool')
    mypixel  = np.zeros(a)
    mypixerr = np.zeros(a)

    for i in range(a):
    
        g     = models.Gaussian1D(amplitude=90.1,mean=np.float(a)/2.,stddev=3.2)
        datad = np.array(data4[:,i])
        wd    = np.array(w[:,i])
        arrd  = np.array(range(len(datad)))
        maskd = (wd > 0.) #(datad >= 0)
        arrd  = arrd[maskd]
        datad = datad[maskd] 
        wd    = wd[maskd]
        err_total = np.sqrt(1./wd)

        try:
            popt, pcov = curve_fit(my_gaussian, arrd, datad,p0=[40,30,3],sigma=err_total)#,sigma=np.sqrt(datad) sigma=np.sqrt(1./wd)**2)
        except:
            mymask[i] = False 
        else:        
            #print(popt)
            #print(np.sqrt(np.diag(pcov)))
            mymu[i]    = popt[1]
            mysigma[i] = my_bootstrap(arrd,datad,err_total)#np.sqrt(np.diag(pcov))[1] 
    
            plt.figure()
            plt.plot(arrd, my_gaussian(arrd, *popt),label='curve fit')
            plt.scatter(arrd,datad,label='data')
            plt.legend()
            plt.savefig('../fig_gauss/'+str(folder)+'/'+str(f[:-5])+'fig_'+str(i)+'.png')
            plt.close()

    

    arr2 = range(a)
    arr2 = np.array(arr2)

    arr2 = arr2[mymask]
    mymu = mymu[mymask]
    mysigma = mysigma[mymask]

    plt.figure()
    plt.xlim(0,60)
    plt.ylim(0,60)     
    plt.imshow(data4, origin='lower', cmap='Greys_r') 
    plt.plot(arr2,mymu,markersize=0.1,lw=1)
    plt.savefig('../figs/'+str(folder)+'/'+str(f[:-5])+'fig5.png')
    plt.close()

    plt.figure()
    plt.xlim(0,60)
    plt.ylim(0,60)
    plt.imshow(data4, origin='lower', cmap='Greys_r') 
    plt.errorbar(arr2,mymu,yerr=mysigma,fmt='o',markersize=0.1,lw=1.0,color='blue')
    plt.savefig('../figs/'+str(folder)+'/'+str(f[:-5])+'fig6.png')
    plt.close()

    mymup = mymu - np.float(b)/2.
    

    x = np.arange(-a/2,a/2,1)
    x = x[mymask]
    

    w = 0.
    w2 = 0.

    mask1 = (x >= 0)*(x < mysize)
    x1 = x[mask1]
    mymup1 = mymup[mask1]
    print('x1 = ',x1)
    print('mymup1 = ',mymup1)

    mask2 = (x >= -mysize)*(x < mysize)
    x2 = x[mask2]
    mymup2 = mymup[mask2]
    print('x2 = ',x2) 
    print('mymup2 = ',mymup2)

    for i in range(len(x1)):
        w += 2.*x1[i]*mymup1[i]/((np.float(mysize))**3)

    for i in range(len(x2)):
        w2 += np.abs(x2[i])*mymup2[i]/((np.float(mysize))**3) 

    w = np.abs(w)
    w2 = np.abs(w2) 
    print(w,w2)

    wf.append(w)
    w2f.append(w2)

vec_s  = np.array(vec_s)
vec_os = np.array(vec_os)

np.savetxt('../output/'+str(folder)+'/'+'size.txt',np.array([vec_s,vec_os]).T)

wf  = np.array(wf)
w2f = np.array(w2f)

np.savetxt('../output/'+str(folder)+'/'+'w_w2.txt',np.array([wf,w2f]).T)

plt.figure()
plt.hist(wf[wf<1])
plt.savefig('../figs/'+str(folder)+'/'+'histw.png')

plt.figure()
plt.hist(w2f[w2f<1])
plt.savefig('../figs/'+str(folder)+'/'+'histw2.png')



















