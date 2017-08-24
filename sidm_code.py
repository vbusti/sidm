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


if __name__=="__main__":

    data_dir = '/home/vinicius/Documents/sidm/data/z_005_02_snr_gt_100_radius_gt_1/i_band'

    folder = 'file_1000'       

    with open(data_dir+"/"+folder+".txt", "r") as ins:
        lfiles = []
        for line in ins:
            lfiles.append(line)

    wrf     = []
    err_wrf = []
    wlf     = []
    err_wlf = []
    wtf     = []
    err_wtf = []
    fil     = []
    type_g  = []

    for f in lfiles:

        fil.append(f[:-6])

        data, weight, size_x = prepare_data(f,data_dir,folder)

        #x,y,err_y,mask = fit_warp_curve(f,data,weight)

        x,y,err_y = np.loadtxt('../output/'+str(folder)+'/'+str(f[:-5])+'warp_curve.txt',unpack=True)
        mask = np.loadtxt('../output/'+str(folder)+'/'+str(f[:-5])+'mask_warp_curve.txt')
        mask = mask.astype(bool)

        w = Warp_Strength(y,err_y,mask,size_x)

        wrs,wr,err_wr = w.calculate_wr()

        wls,wl,err_wl = w.calculate_wl()

        wt,err_wt = w.calculate_wt()

        nsig = 3.

        if((wrs >  nsig*err_wr)*( wls >  nsig*err_wl)): type_g.append('U')
        if((wrs < -nsig*err_wr)*( wls < -nsig*err_wl)): type_g.append('U')
        if((wrs >  nsig*err_wr)*( wls < -nsig*err_wl)): type_g.append('S')
        if((wrs < -nsig*err_wr)*( wls >  nsig*err_wl)): type_g.append('S')
        if((wr  >  nsig*err_wr)*( wl  <  nsig*err_wl)): type_g.append('NR')
        if((wr  <  nsig*err_wr)*( wl  >  nsig*err_wl)): type_g.append('NL')
        if((wr  <  nsig*err_wr)*( wl  <  nsig*err_wl)): type_g.append('F') 
    
        wrf.append(wr)
        err_wrf.append(err_wr)
        wlf.append(wl)
        err_wlf.append(err_wl)
        wtf.append(wt)
        err_wtf.append(err_wt)


    fil = np.array(fil)
    type_g = np.array(type_g)

    wrf     = np.array(wrf)
    err_wrf = np.array(err_wrf)
    wlf     = np.array(wlf)
    err_wlf = np.array(err_wlf)
    wtf     = np.array(wtf)
    err_wtf = np.array(err_wtf)

    final = np.column_stack((fil,wrf,err_wrf,wlf,err_wlf,wtf,err_wtf,type_g))
    #np.savetxt('../output/'+str(folder)+'/'+'wr_wl_wt_3sigma.txt',final,delimiter=" ",fmt="%s")
    
    plt.figure()
    plt.hist(wrf)
    plt.savefig('../figs/'+str(folder)+'/'+'histwr.png')

    plt.figure()
    plt.hist(err_wrf)
    plt.savefig('../figs/'+str(folder)+'/'+'hist_err_wr.png')

    plt.figure()
    plt.hist(err_wrf/wrf)
    plt.savefig('../figs/'+str(folder)+'/'+'ratio_hist_wr.png')

    plt.figure()
    plt.hist(wlf)
    plt.savefig('../figs/'+str(folder)+'/'+'histwl.png')

    plt.figure()
    plt.hist(err_wlf)
    plt.savefig('../figs/'+str(folder)+'/'+'hist_err_wl.png')

    plt.figure()
    plt.hist(err_wlf/wlf)
    plt.savefig('../figs/'+str(folder)+'/'+'ratio_hist_wl.png')

    plt.figure()
    plt.hist(wtf)
    plt.savefig('../figs/'+str(folder)+'/'+'histwt.png')

    plt.figure()
    plt.hist(err_wtf)
    plt.savefig('../figs/'+str(folder)+'/'+'hist_err_wt.png')

    plt.figure()
    plt.hist(err_wtf/wtf)
    plt.savefig('../figs/'+str(folder)+'/'+'ratio_hist_wt.png')




















