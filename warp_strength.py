#!/usr/bin/env python
# encoding: UTF8
#

from __future__ import print_function
import numpy as np
import random
from gapp import dgp

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
    def __init__(self,y,err_y,mask,size_x):
        self.y      = y
        self.err_y  = err_y
        self.mask   = mask
        self.size_x = size_x

    def w1(self): 
        """
        calculate w1= ...
        """ 
        y, err_y, mask, size_x = self.y, self.err_y, self.mask, self.size_x
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

    def calculate_w2(self):
        """
        calculate w2= ...
        """ 
        y, err_y, mask, size_x = self.y, self.err_y, self.mask, self.size_x
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

    def calculate_w3(self):
        """
        calculate w3= ...
        """
        y, err_y, mask, size_x = self.y, self.err_y, self.mask, self.size_x 
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

    def calculate_w4(self):
        """
        calculate w4= ...
        """
        y, err_y, mask, size_x = self.y, self.err_y, self.mask, self.size_x 
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

    def calculate_w5(self):
        """
        calculate w5= ...
        """
        y, err_y, mask, size_x = self.y, self.err_y, self.mask, self.size_x 
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

        err_w = boot_err_w(x1,y1,err_y1)

        return np.abs(w)/np.sum(1./err_y1**2),err_w


    def calculate_w_gp(self):
        """
        calculate w_gp= ...
        """
        y, err_y, mask, size_x = self.y, self.err_y, self.mask, self.size_x 
        x = np.array(np.arange(-len(mask)/2,len(mask)/2,1))
        x = x[mask]
        w = 0.
        err_w = 0.

        mask1 = (x >= 0)*(x < size_x)#*(err_y <= 5)# np.percentile(err_y,90.))  
        x1 = x[mask1]
        y1 = y[mask1]
        err_y1 = err_y[mask1]

        x1 = np.array(x1)
        y1 = np.array(range(len(x1)))#np.array(y1)
        err_y1 = np.ones(len(x1))#np.array(err_y1)

        print('x_gp=',len(x1),len(y1),len(err_y1))

        # nstar points of the function will be reconstructed 
        # between xmin and xmax
        xmin = 0.0
        xmax = len(x1)
        nstar = len(x1) + 1

        # initial values of the hyperparameters
        initheta = [10.5, 10.5]

        # initialization of the Gaussian Process
        g = dgp.DGaussianProcess(x1, y1, err_y1, cXstar=(xmin, xmax, nstar))

        # training of the hyperparameters and reconstruction of the function
        #(rec, theta) = g.gp(theta=initheta)

        '''
        for i in range(len(x1)):
            w += 2.*rec[0,i]*rec[1,i]/((np.float(np.max(x1)))**3)
            err_w += (2.*rec[0,i]*rec[2,i]/((np.float(np.max(x1)))**3))**2
        '''
        return 0,0#np.abs(w),err_w

    def calculate_wr(self):
        """
        calculate w5= ...
        """
        y, err_y, mask, size_x = self.y, self.err_y, self.mask, self.size_x 
        x = np.array(np.arange(-len(mask)/2,len(mask)/2,1))
        x = x[mask]
        w = 0.
        err_w = 0.

        mask1 = (x >= 0)*(x < size_x)*(err_y <= 5)  
        x1 = x[mask1]
        y1 = y[mask1]
        err_y1 = err_y[mask1]
    
        for i in range(len(x1)):
            w += np.abs(x1[i])*y1[i]/err_y1[i]**2/((np.float(size_x))**3)

        err_w = boot_err_w(x1,y1,err_y1,size_x)

        return w/np.sum(1./err_y1**2),np.abs(w)/np.sum(1./err_y1**2),err_w


    def calculate_wl(self):
        """
        calculate wl= ...
        """
        y, err_y, mask, size_x = self.y, self.err_y, self.mask, self.size_x 
        x = np.array(np.arange(-len(mask)/2,len(mask)/2,1))
        x = x[mask]
        w = 0.
        err_w = 0.

        mask1 = (x >= -size_x)*(x < 0)*(err_y <= 5)  
        x1 = x[mask1]
        y1 = y[mask1]
        err_y1 = err_y[mask1]
    
        for i in range(len(x1)):
            w += np.abs(x1[i])*y1[i]/err_y1[i]**2/((np.float(size_x))**3)

        err_w = boot_err_w(x1,y1,err_y1,size_x)

        return w/np.sum(1./err_y1**2),np.abs(w)/np.sum(1./err_y1**2),err_w

    def calculate_wt(self):
        """
        calculate wt= ...
        """
        y, err_y, mask, size_x = self.y, self.err_y, self.mask, self.size_x 
        x = np.array(np.arange(-len(mask)/2,len(mask)/2,1))
        x = x[mask]
        w = 0.
        err_w = 0.

        mask1 = (x >= 0)*(x < size_x)*(err_y <= 5)  
        x1 = x[mask1]
        y1 = y[mask1]
        err_y1 = err_y[mask1]
 
        for i in range(len(x1)):
            w += np.abs(x1[i])*y1[i]/err_y1[i]**2/((np.float(size_x))**3)

        err_w = boot_err_w(x1,y1,err_y1,2*size_x)

        return np.abs(w)/np.sum(1./err_y1**2),err_w #w/np.sum(1./err_y1**2),
  
