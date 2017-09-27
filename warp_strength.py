#!/usr/bin/env python
# encoding: UTF8
#

from __future__ import print_function
import numpy as np
import random

def boot_err_w(x,y,err_y,size_x):
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
            w_sum += np.abs(x_b[i])*y_b[i]/err_yb[i]**2/((np.float(size_x))**3)
        mean_values[i] = w_sum/np.sum(1./err_yb**2)
    return np.std(mean_values)


class Warp_Strength:
    "It calculates the warp strengh using different definitions"
    def __init__(self,y,err_y,mask,size_x,xcent,ycent,x_shape,y_shape):
        self.y       = y
        self.err_y   = err_y
        self.mask    = mask
        self.size_x  = size_x
        self.xcent   = xcent
        self.ycent   = ycent
        self.x_shape = x_shape
        self.y_shape = y_shape

    def calculate_wr(self):
        """
        calculate w using the right part of the galaxy weighted by 1/sigma**2
        """
        y, err_y, mask, size_x, xcent, ycent, x_shape = self.y, self.err_y, self.mask, self.size_x, self.xcent, self.ycent, self.x_shape 
        x = np.array(np.arange(-len(mask)/2,len(mask)/2,1))
        x = x[mask]
        x = x - (xcent - (x_shape/2. -1.))
        w = 0.
        err_w = 0.

        mask1 = (x >= 0)*(x < size_x)   
        x1 = x[mask1]
        y1 = y[mask1]
        err_y1 = err_y[mask1]
    
        for i in range(len(x1)):
            w += np.abs(x1[i])*y1[i]/err_y1[i]**2/((np.float(size_x))**3)
            err_w += np.abs(x1[i])**2/err_y1[i]**2/((np.float(size_x))**6)

        err_w = np.sqrt(err_w/np.sum(1./err_y1**2)**2)

        err_w2 = boot_err_w(x1,y1,err_y1,size_x)
        print('err_r',err_w,err_w2) 

        return w/np.sum(1./err_y1**2),np.abs(w)/np.sum(1./err_y1**2),err_w


    def calculate_wl(self):
        """
        calculate w using the left part of the galaxy weighted by 1/sigma**2
        """
        y, err_y, mask, size_x, xcent, ycent, x_shape = self.y, self.err_y, self.mask, self.size_x, self.xcent, self.ycent, self.x_shape
        x = np.array(np.arange(-len(mask)/2,len(mask)/2,1))
        x = x[mask]
        x = x - (xcent - (x_shape/2. -1.))
        w = 0.
        err_w = 0.

        mask1 = (x >= -size_x)*(x < 0)  
        x1 = x[mask1]
        y1 = y[mask1]
        err_y1 = err_y[mask1]
    
        for i in range(len(x1)):
            w += np.abs(x1[i])*y1[i]/err_y1[i]**2/((np.float(size_x))**3)
            err_w += np.abs(x1[i])**2/err_y1[i]**2/((np.float(size_x))**6)

        err_w = np.sqrt(err_w/np.sum(1./err_y1**2)**2)

        err_w2 = boot_err_w(x1,y1,err_y1,size_x)
        print('err_l',err_w,err_w2)

        #err_w = boot_err_w(x1,y1,err_y1,size_x)

        return w/np.sum(1./err_y1**2),np.abs(w)/np.sum(1./err_y1**2),err_w


    def calculate_wt(self):
        """
        calculate w using the whole galaxy weighted by 1/sigma**2
        """
        y, err_y, mask, size_x, xcent, ycent, x_shape = self.y, self.err_y, self.mask, self.size_x, self.xcent, self.ycent, self.x_shape
        x = np.array(np.arange(-len(mask)/2,len(mask)/2,1))
        x = x[mask]
        x = x - (xcent - (x_shape/2. -1.))
        w = 0.
        err_w = 0.

        mask1 = (x >= 0)*(x < size_x)  
        x1 = x[mask1]
        y1 = y[mask1]
        err_y1 = err_y[mask1]
 
        for i in range(len(x1)):
            w += np.abs(x1[i])*y1[i]/err_y1[i]**2/((np.float(size_x))**3)
            err_w += np.abs(x1[i])**2/err_y1[i]**2/((np.float(size_x))**6)

        err_w = np.sqrt(err_w/np.sum(1./err_y1**2)**2)

        err_w2 = boot_err_w(x1,y1,err_y1,size_x)
        print('err t',err_w,err_w2)

        #err_w = boot_err_w(x1,y1,err_y1,2*size_x)

        return np.abs(w)/np.sum(1./err_y1**2),err_w #w/np.sum(1./err_y1**2),

    


  
