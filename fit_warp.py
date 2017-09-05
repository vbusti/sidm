#!/usr/bin/env python
# encoding: UTF8
#

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
import random


def my_gaussian(x,amp,mean,sigma):
    return amp*np.exp(-(x-mean)**2/(2.*sigma**2)) 

def my_bootstrap(x,y,sigma):
    acc=0
    mean_values = np.zeros(500)
    while(acc < 500):
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
            

def fit_warp_curve(file,folder,data,weight,ycent):
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
    
        datad = np.array(data[:,i])
        wd    = np.array(weight[:,i])
        arrd  = np.array(range(len(datad)))
        maskd = (wd > 0.) 
        arrd  = arrd[maskd]
        datad = datad[maskd] 
        wd    = wd[maskd]
        err_total = np.sqrt(1./wd)
        #print(arrd)
        #print(datad)
        #print(wd)

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
    mymup = mymu - np.float(b)/2. - (ycent - 45.)

    plt.figure()
    plt.xlim(10,50)
    plt.ylim(20,40)     
    plt.imshow(data, origin='lower', cmap='Greys_r') 
    plt.plot(arr2,mymu,markersize=0.1,lw=1)
    plt.savefig('../figs/'+str(folder)+'/'+str(file[:-5])+'fig5.png')
    plt.close()

    plt.figure()
    plt.xlim(10,50)
    plt.ylim(20,40) 
    plt.imshow(data, origin='lower', cmap='Greys_r') 
    plt.errorbar(arr2,mymu,yerr=mysigma,fmt='o',markersize=0.1,lw=1.0,color='blue')
    plt.savefig('../figs/'+str(folder)+'/'+str(file[:-5])+'fig6.png')
    plt.close()

    np.savetxt('../output/'+str(folder)+'/'+str(file[:-5])+'warp_curve.txt',np.array([arr2,mymup,mysigma]).T)
    np.savetxt('../output/'+str(folder)+'/'+str(file[:-5])+'mask_warp_curve.txt',np.array([mymask]).T)

    return arr2, mymup, mysigma, mymask

