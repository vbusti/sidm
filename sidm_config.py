#!/usr/bin/env python
# encoding: UTF8
#

# This is my configuration file. It will eventually mature to something more sophisticated, but so far I don't want to mess up anything


# data_dir points to the input catalog, folder is the name used for the output files

data_dir = '/home/vinicius/Documents/sidm/data/z_005_020_snr_10_50_radius_gt_1/v2/i_band'

folder = 'snr_10_50_r_gt_1'  

# number of processes

nprocesses = 4 

# below I should say in which way the code will be run (FALSE means the first steps were run previously, otherwise the code it will fail)

RUN_PREPARE       = True # this must be true for the moment, although this is not time consuming at all
RUN_FIT_WARP      = True
RUN_WARP_STRENGTH = True

# plots
PLOT_GAUSSIAN_COLUMNS = False

# real data or simulation
# real data demands a weight map, simulation estimates the noise from the sky level

DATA_TYPE = 'REAL' # 'SIMULATION'

# I'll also have an option for plotting final results



















