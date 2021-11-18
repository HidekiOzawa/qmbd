#%%
# -*- coding: utf-8 -*-
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
#from scipy.ndimage import gaussian_filter1d

import sys

sys.path.append("C:\\Users\\Lab\\Desktop\\Python_code\\func")

import generate_cmap as gcm
from fit_func import *
import set_default_params

# constants
s_px = 2 * 6.45 # pixel size of PCO CCD camera (um/px)
gexp = 9.797901 # expected gravity acceleration (m/s^2)
hbar = 1.054571726e-34 # J/Hz
kB = 1.38064852e-23 # J/K
mass = 86.909180520 * 1.660538782e-27 # kg
sigma0 = 2.909 * 1e-9 * 1e8 # 
# %%
dataNo = 3
scanNo = 6
figsave = True
# %%
path = r"C:\\Users\\Lab\\Desktop\\Python_code\\20211115"
path_img = path + r"\\data" + str('%03d' % dataNo) + "\\raw"
path_scan = path + r"\\scandir\\scan" + str('%03d' % scanNo) + '.txt'
path_param = path + r"\\scandir\\Parameters" + str('%03d' % scanNo) + '.txt'

scan_data = pd.read_csv(path_scan, delimiter='\t')
# %%
# Analysis area
#x0 = 151; x1 = 525
#y0 = 91; y1 = 500
x0 = 1; x1 = 127
y0 = 1; y1 = 127

width = x1- x0
height = y1- y0

# %%
share = Path(path_img)
if share.exists():
    TOFtime = scan_data['T0008: TOF time (msec.)']
    
    trans_files = list(Path(path_img).glob('seq*_trans001.tiff'))
    flat_files = list(Path(path_img).glob('seq*_flat001.tiff'))
    update_files = list(Path(path_img).glob('seq*_update001.tiff'))
    TOFtimes = []
    params = []; params_err = []
    cx0 = []; cy0 = []
    cx0_err = []; cy0_err = []
    wx0 = []; wy0 = []
    wx0_err = []; wy0_err = []
    
    OD_sum = []
    img_sub_ave = np.zeros([height, width], dtype=np.float64)
    
    for N in range(len(trans_files)):
        img_trans = np.array(Image.open(trans_files[N]))
        img_flat = np.array(Image.open(flat_files[N]))
        img_update = np.array(Image.open(update_files[N]))
        img = np.zeros(img_trans.shape)
        
        img_sub = img[y0:y1, x0:x1]
        
        mask_bg = np.zeros(img_sub.shape)
        mask_img = np.zeros(img_sub.shape)
#        sm = 10
#        mask_bg[:sm, :] = 1
##        mask_bg[:, :sm] = 1
##        mask_bg[:, -sm:] = 1
#        mask_bg[-sm:, :] = 1
        mask_bg[int(height/2)-6:int(height/2)+9, int(width/2)-13:int(width/2)+6] = 1
        mask_bg[int(height/2)-3:int(height/2)+6, :] = 0
        mask_img[int(height/2)-3:int(height/2)+6, int(width/2)-10:int(width/2)+3] = 1
#        mask_bg[mask_bg==0] = 1
#        mask_bg[int(width/2)-5:int(width/2)+5, :] = 1
        
        bg_count = np.sum(img_sub * mask_bg) / np.sum(mask_bg)
        img_sub = img_sub - bg_count
        
# %%
print(img_flat)

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.pcolormesh(img_trans, vmin=0, vmax=5000, shading='auto')
ax.set_aspect('equal')

ax.set_xticks([])
ax.set_yticks([])


plt.show()
# %%
sweep_data = pd.read_csv(path_scan, sep='\t')

seq_index = sweep_data['#No.']
ts_TOF = sweep_data['T0046: TOF in the vertical lattice (usec.)']

list_t_TOF = ts_TOF.unique()

seq_index = np.array(seq_index)
ts_TOF = np.array(ts_TOF)
# %%
print(ts_TOF)
# %%
