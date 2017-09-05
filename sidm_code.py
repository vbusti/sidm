#!/usr/bin/env python
# encoding: UTF8
#

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from warp_strength import *
from prep_data import *
from fit_warp import *
from multiprocessing import Pool

def calc_all_steps(f):

    data_dir = '/home/vinicius/Documents/sidm/data/z_005_02_snr_gt_100_radius_gt_1/i_band'

    folder = 'file_1000'       

    data, weight, size_x, xcent, ycent = prepare_data(f,data_dir,folder)

    x,y,err_y,mask = fit_warp_curve(f,folder,data,weight,ycent)

    #x,y,err_y = np.loadtxt('../output/'+str(folder)+'/'+str(f[:-5])+'warp_curve.txt',unpack=True)
    #mask = np.loadtxt('../output/'+str(folder)+'/'+str(f[:-5])+'mask_warp_curve.txt')
    #mask = mask.astype(bool)

    w = Warp_Strength(y,err_y,mask,size_x,xcent,ycent)

    wrs,wr,err_wr = w.calculate_wr()

    wls,wl,err_wl = w.calculate_wl()

    wt,err_wt = w.calculate_wt()

    nsig = 3.

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

    data_dir = '/home/vinicius/Documents/sidm/data/z_005_02_snr_gt_100_radius_gt_1/i_band'

    folder = 'file_1000'       
    
    with open(data_dir+"/"+folder+".txt", "r") as ins: #open("my_files.txt", "r") as ins:
        lfiles = []
        for line in ins:
            lfiles.append(line)

    p = Pool()
    result = p.map(calc_all_steps,lfiles)
    p.close()
    p.join()
    result = np.array(result,dtype='f8,f8,f8,f8,f8,f8,f8,f8,|S1,|S29')

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

    final = np.column_stack((fil,wr,err_wr,wl,err_wl,wt,err_wt))#,type_g))
    np.savetxt('../output/'+str(folder)+'/'+'wr_wl_wt_3sigma.txt',final,delimiter=" ",fmt="%s")
    np.savetxt('../output/'+str(folder)+'/'+'gtype_3sigma.txt',type_g,fmt="%s")


    mask = wr < 0.2

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




















