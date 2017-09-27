#!/usr/bin/env python
# encoding: UTF8
#

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sidm_config as cfg
from matplotlib import cm
from warp_strength import *
from prep_data import *
from fit_warp import *
from multiprocessing import Pool
from gal_sim import *

def calc_all_steps(f):


    data_dir = cfg.data_dir

    folder = cfg.folder   

    print(f)
        
    try:     

        data, weight, size_x, xcent, ycent, x_shape, y_shape = prepare_data(f,data_dir,folder)


        if(cfg.RUN_FIT_WARP == True):
            x,y,err_y,mask = fit_warp_curve(f,folder,data,weight,ycent,x_shape,y_shape)
        else:
            pars = np.loadtxt('../output/'+str(folder)+'/'+str(f[:-5])+'_size_xcent_ycent_xy_shape.txt',unpack=True)
            size_x  = pars[0]
            xcent   = pars[1]
            ycent   = pars[2]
            x_shape = pars[3]
            y_shape = pars[4]

            x,y,err_y = np.loadtxt('../output/'+str(folder)+'/'+str(f[:-5])+'warp_curve.txt',unpack=True)
            mask = np.loadtxt('../output/'+str(folder)+'/'+str(f[:-5])+'mask_warp_curve.txt')
            mask = mask.astype(bool)

        if(cfg.RUN_WARP_STRENGTH == True):
            w = Warp_Strength(y,err_y,mask,size_x,xcent,ycent,x_shape,y_shape)
            wrs,wr,err_wr = w.calculate_wr()
            wls,wl,err_wl = w.calculate_wl()
            wt,err_wt = w.calculate_wt()
        else:
            fil,wr,err_wr,wl,err_wl,wt,err_wt = np.loadtxt('../output/'+str(folder)+'/'+'wr_wl_wt_5sigma.txt',delimiter=" ",fmt="%s",unpack=True)
            wrs    = np.loadtxt('../output/'+str(folder)+'/'+'wrs_5sigma.txt')
            type_g = np.loadtxt('../output/'+str(folder)+'/'+'gtype_5sigma.txt',fmt="%s")  
    

    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, f[:-1]

    else:
        
        nsig = 5

        
        if((wrs >  nsig*err_wr)*( wls >  nsig*err_wl)):
            type_g = 'U'
        elif((wrs < -nsig*err_wr)*( wls < -nsig*err_wl)): 
            type_g = 'U'
        elif((wrs >  nsig*err_wr)*( wls < -nsig*err_wl)): 
            type_g = 'S'
        elif((wrs < -nsig*err_wr)*( wls >  nsig*err_wl)): 
            type_g = 'S'
        elif((wr  >  nsig*err_wr)*( wl  <  nsig*err_wl)): 
            type_g = 'R'
        elif((wr  <  nsig*err_wr)*( wl  >  nsig*err_wl)): 
            type_g = 'L'
        elif((wr  <  nsig*err_wr)*( wl  <  nsig*err_wl)): 
            type_g = 'F'
        else:
            type_g = "W" 
        
    
    return wrs, wr, err_wr, wls, wl, err_wl, wt, err_wt, type_g, f[:-1]


if __name__=="__main__":

    data_dir = cfg.data_dir

    folder = cfg.folder   
        
    with open(data_dir+"/"+folder+".txt", "r") as ins: #open("my_files.txt", "r") as ins:
        lfiles = []
        for line in ins:
            lfiles.append(line)
        
    p = Pool(cfg.nprocesses)
    result = p.map(calc_all_steps,lfiles)
    print('result = ',result)
    p.close()
    p.join()
    result = np.array(result,dtype='f8,f8,f8,f8,f8,f8,f8,f8,|S1,|S12')

    wrs    = result['f0']
    wr     = result['f1']
    err_wr = result['f2']
    wls    = result['f3']
    wl     = result['f4']
    err_wl = result['f5']
    wt     = result['f6']
    err_wt = result['f7']
    type_g = result['f8'] 
    fil    = result['f9']

    if(cfg.RUN_WARP_STRENGTH == True):   
        final = np.column_stack((fil,wr,err_wr,wl,err_wl,wt,err_wt))#,type_g))
        np.savetxt('../output/'+str(folder)+'/'+'wr_wl_wt_5sigma.txt',final,delimiter=" ",fmt="%s")
        np.savetxt('../output/'+str(folder)+'/'+'gtype_5sigma.txt',type_g,fmt="%s")
        np.savetxt('../output/'+str(folder)+'/'+'wrs_5sigma.txt',wrs,delimiter=" ",fmt="%s")

    mask_aux = np.zeros(len(wr))

    for i in range(len(wr)):
        if((not np.isnan(wr[i]))*(not np.isnan(wl[i]))*(not np.isnan(wt[i]))*(not np.isnan(wrs[i]))):
            mask_aux[i] = 1 
   

    mask = (mask_aux == 1)#*(wr < 0.2)

    wr = wr[mask]
    wl = wl[mask]
    wt = wt[mask]

    err_wr = err_wr[mask]
    err_wl = err_wl[mask]
    err_wt = err_wt[mask]


    plt.figure()
    plt.hist(wr)
    plt.savefig('../figs/'+str(folder)+'/'+'histwr.png')

    plt.figure()
    plt.hist(err_wr)
    plt.savefig('../figs/'+str(folder)+'/'+'hist_err_wr.png')

    plt.figure()
    plt.hist(err_wr/wr)
    plt.savefig('../figs/'+str(folder)+'/'+'ratio_hist_wr.png')

    plt.figure()
    plt.hist(wl)
    plt.savefig('../figs/'+str(folder)+'/'+'histwl.png')

    plt.figure()
    plt.hist(err_wl)
    plt.savefig('../figs/'+str(folder)+'/'+'hist_err_wl.png')

    plt.figure()
    plt.hist(err_wl/wl)
    plt.savefig('../figs/'+str(folder)+'/'+'ratio_hist_wl.png')

    plt.figure()
    plt.hist(wt)
    plt.savefig('../figs/'+str(folder)+'/'+'histwt.png')

    plt.figure()
    plt.hist(err_wt)
    plt.savefig('../figs/'+str(folder)+'/'+'hist_err_wt.png')

    plt.figure()
    plt.hist(err_wt/wt)
    plt.savefig('../figs/'+str(folder)+'/'+'ratio_hist_wt.png')
    




















