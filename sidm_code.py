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

# general functions

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

def boot_err_w(x,y,err_y):
    mean_values = np.zeros(1000)
    for i in range(1000):
        x_b    = np.zeros(len(x))
        y_b    = np.zeros(len(x))
        err_yb = np.zeros(len(x)) 
        for i in range(len(x)):
            aux=random.randint(0,len(x)-1)
            x_b[i]    = x[aux] 
            y_b[i]    = y[aux]
            err_yb[i] = err_y[aux] 
        w_sum = 0.
        for i in range(len(x)):
            w_sum += 2.*x_b[i]*y_b[i]/err_yb[i]**2/((np.float(np.max(x)))**3)
        mean_values[i] = w_sum/np.sum(1./err_yb**2)
    return np.std(mean_values)

#########################
            

def prepare_data(file):
    """
    prepare_data picks a file with the image of the galaxy, detect the central object, rotate it to the major axis, and returns 
    the data and errors ready to fit a warp curve, along with the maximum distance from the center 

    """
    hdu = fits.open('/home/vinicius/Documents/sidm/data/i_band/'+str(file[:-1]))[0]
    wcs = WCS(hdu.header)

    data = hdu.data

    weight = fits.open('/home/vinicius/Documents/sidm/data/i_band/'+str(file[:-1]))[1].data

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

    data4 = rotate(data, np.rad2deg(mytheta))
    data4 = data4[data4.shape[0]/2 - 30:data4.shape[0]/2 + 30,data4.shape[1]/2 - 30:data4.shape[1]/2 + 30]   


    w_rot  = rotate(weight, np.rad2deg(mytheta))
    w      = w_rot[w_rot.shape[0]/2 - 30:w_rot.shape[0]/2 + 30,w_rot.shape[1]/2 - 30:w_rot.shape[1]/2 + 30] 

    plt.figure()    
    plt.imshow(data4, origin='lower', cmap='Greys_r') 
    plt.savefig('../figs/'+str(folder)+'/'+str(f[:-5])+'fig4.png')

    return data4, w, mysize 


def fit_warp_curve(file,data,weight):
    """
    fit_warp_curve picks data and errors to fit the warp curve.
    file is used only for plotting 
    It returns the warp curve: (x,y,err_y) and a mask to calculate the warpness
    """
    a = data.shape[1]
    b = data.shape[0]

    mymu     = np.zeros(a)
    mysigma  = np.zeros(a)
    mymask   = np.ones(a,dtype='bool')

    for i in range(a):
    
        g     = models.Gaussian1D(amplitude=90.1,mean=np.float(a)/2.,stddev=3.2)
        datad = np.array(data[:,i])
        wd    = np.array(weight[:,i])
        arrd  = np.array(range(len(datad)))
        maskd = (wd > 0.) 
        arrd  = arrd[maskd]
        datad = datad[maskd] 
        wd    = wd[maskd]
        err_total = np.sqrt(1./wd)

        try:
            popt, pcov = curve_fit(my_gaussian, arrd, datad,p0=[40,30,3],sigma=err_total)
        except:
            mymask[i] = False 
        else:        
            mymu[i]    = popt[1]
            mysigma[i] = my_bootstrap(arrd,datad,err_total) 
    
            plt.figure()
            plt.plot(arrd, my_gaussian(arrd, *popt),label='curve fit')
            plt.scatter(arrd,datad,label='data')
            plt.legend()
            plt.savefig('../fig_gauss/'+str(folder)+'/'+str(file[:-5])+'fig_'+str(i)+'.png')
            plt.close()
    
    arr2 = range(a)
    arr2 = np.array(arr2)

    arr2 = arr2[mymask]
    mymu = mymu[mymask]
    mysigma = mysigma[mymask]
    mymup = mymu - np.float(b)/2.

    plt.figure()
    plt.xlim(0,60)
    plt.ylim(0,60)     
    plt.imshow(data, origin='lower', cmap='Greys_r') 
    plt.plot(arr2,mymu,markersize=0.1,lw=1)
    plt.savefig('../figs/'+str(folder)+'/'+str(file[:-5])+'fig5.png')
    plt.close()

    plt.figure()
    plt.xlim(0,60)
    plt.ylim(0,60)
    plt.imshow(data, origin='lower', cmap='Greys_r') 
    plt.errorbar(arr2,mymu,yerr=mysigma,fmt='o',markersize=0.1,lw=1.0,color='blue')
    plt.savefig('../figs/'+str(folder)+'/'+str(file[:-5])+'fig6.png')
    plt.close()

    np.savetxt('../output/'+str(folder)+'/'+str(file[:-5])+'warp_curve.txt',np.array([arr2,mymup,mysigma]).T)
    np.savetxt('../output/'+str(folder)+'/'+str(file[:-5])+'mask_warp_curve.txt',np.array([mymask]).T)

    return arr2, mymup, mysigma, mymask

def calculate_w1(y,err_y,mask,size_x):
    """
    calculate w1= ...
    """ 
    x = np.array(np.arange(-len(mask)/2,len(mask)/2,1))
    x = x[mask]
    w = 0.
    err_w = 0.

    mask1 = (x >= 0)*(x < size_x)  
    x1 = x[mask1]
    y1 = y[mask1]
    err_y1 = err_y[mask1]

    for i in range(len(x1)):
        w += 2.*x1[i]*y1[i]/((np.float(size_x))**3)
        err_w += (2.*x1[i]*err_y1[i]/((np.float(size_x))**3))**2

    return np.abs(w),np.sqrt(err_w)

def calculate_w2(y,err_y,mask,size_x):
    """
    calculate w2= ...
    """ 
    x = np.arange(-len(mask)/2,len(mask)/2,1)
    x = x[mask]
    w = 0.
    err_w = 0.

    mask1 = (x >= -size_x)*(x < size_x)
    x1 = x[mask1]
    y1 = y[mask1]
    err_y1 = err_y[mask1]

    for i in range(len(x1)):
        w += np.abs(x1[i])*y1[i]/((np.float(size_x))**3) 
        err_w += (np.abs(x1[i])*err_y1[i]/((np.float(size_x))**3))**2

    return np.abs(w),np.sqrt(err_w)

def calculate_w3(y,err_y,mask,size_x):
    """
    calculate w3= ...
    """ 
    x = np.array(np.arange(-len(mask)/2,len(mask)/2,1))
    x = x[mask]
    w = 0.
    err_w = 0.

    mask1 = (x >= 0)*(x < size_x)  
    x1 = x[mask1]
    y1 = y[mask1]
    err_y1 = err_y[mask1]

    for i in range(len(x1)):
        w += 2.*x1[i]*y1[i]/(err_y1[i]**2*(np.float(size_x))**3)
        err_w += (2.*x1[i]*err_y1[i]/((np.float(size_x))**3))**2

    return np.abs(w)/np.sum(1./err_y1**2),np.sqrt(err_w)

# using sigma-clipping to remove huge error-bars

def calculate_w4(y,err_y,mask,size_x):
    """
    calculate w4= ...
    """ 
    x = np.array(np.arange(-len(mask)/2,len(mask)/2,1))
    x = x[mask]
    w = 0.
    err_w = 0.


    mask1 = (x >= 0)*(x < size_x)*(err_y <= 5)# np.percentile(err_y,90.))  
    x1 = x[mask1]
    y1 = y[mask1]
    err_y1 = err_y[mask1]

    print('x1=',np.max(x1),size_x)

    plt.figure()
    plt.hist(err_y1,bins=10)
    plt.savefig('../figs/'+str(folder)+'/temp/'+str(f[:-5])+'erry.png')
    plt.close()

    
    for i in range(len(x1)):
        w += 2.*x1[i]*y1[i]/((np.float(np.max(x1)))**3)
        err_w += (2.*x1[i]*err_y1[i]/((np.float(np.max(x1)))**3))**2

    return np.abs(w),np.sqrt(err_w)


# cutting errors and weighting

def calculate_w5(y,err_y,mask,size_x):
    """
    calculate w5= ...
    """ 
    x = np.array(np.arange(-len(mask)/2,len(mask)/2,1))
    x = x[mask]
    w = 0.
    err_w = 0.


    mask1 = (x >= 0)*(x < size_x)*(err_y <= 5)# np.percentile(err_y,90.))  
    x1 = x[mask1]
    y1 = y[mask1]
    err_y1 = err_y[mask1]

    print('x1=',np.max(x1),size_x)

    
    for i in range(len(x1)):
        w += 2.*x1[i]*y1[i]/err_y1[i]**2/((np.float(np.max(x1)))**3)
        #err_w += (2.*x1[i]*err_y1[i]/((np.float(np.max(x1)))**3))**2

    err_w = boot_err_w(x1,y1,err_y1)

    return np.abs(w)/np.sum(1./err_y1**2),err_w



       

with open("my_files.txt", "r") as ins:
    lfiles = []
    for line in ins:
        lfiles.append(line)


folder = 'temp'

wf    = []
w2f   = []
errw  = []
errw2 = []
w3f   = []
errw3 = []
w4f   = []
errw4 = []
w5f   = []
errw5 = []
fil   = []




for f in lfiles:

    fil.append(f[:-6])

    data, weight, size_x = prepare_data(f)

    #x,y,err_y,mask = fit_warp_curve(f,data,weight)

    x,y,err_y = np.loadtxt('../output/'+str(folder)+'/'+str(f[:-5])+'warp_curve.txt',unpack=True)
    mask = np.loadtxt('../output/'+str(folder)+'/'+str(f[:-5])+'mask_warp_curve.txt')
    mask = mask.astype(bool)
    

    w1,err1 = calculate_w1(y,err_y,mask,size_x)

    w2,err2 = calculate_w2(y,err_y,mask,size_x)

    w3,err3 = calculate_w3(y,err_y,mask,size_x)

    w4,err4 = calculate_w4(y,err_y,mask,size_x)

    w5,err5 = calculate_w5(y,err_y,mask,size_x)
       
    print(w1,err1,w2,err2,w3,w4,err4,w5,err5)

    wf.append(w1)
    w2f.append(w2)
    errw.append(err1)
    errw2.append(err2) 
    w3f.append(w3)
    w4f.append(w4)
    errw4.append(err4)
    w5f.append(w5)
    errw5.append(err5)


fil = np.array(fil)

wf    = np.array(wf)
w2f   = np.array(w2f)
errw  = np.array(errw)
errw2 = np.array(errw2) 
w3f   = np.array(w3f)
w4f   = np.array(w4f)
errw4 = np.array(errw4)
w5f   = np.array(w5f)
errw5 = np.array(errw5)


#final = np.column_stack((fil,wf,errw,w2f,errw2,w3f,w4f,errw4))
final = np.column_stack((fil,wf,errw,w4f,errw4,w5f,errw5))
np.savetxt('../output/'+str(folder)+'/'+'w_w2_err.txt',final,delimiter=" ",fmt="%s")


#np.savetxt('../output/'+str(folder)+'/'+'w_w2_err.txt',np.array([fil,wf,errw,w2f,errw2]).T)


plt.figure()
plt.hist(w4f[errw4 < 0.3])
plt.savefig('../figs/'+str(folder)+'/'+'histw_err_lt_5.png')

plt.figure()
plt.hist(errw4[errw4 < 0.3])
plt.savefig('../figs/'+str(folder)+'/'+'hist_errw_lt_5.png')


'''
plt.figure()
plt.hist(w2f[w2f<1])
plt.savefig('../figs/'+str(folder)+'/'+'histw2.png')
'''


















