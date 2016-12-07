# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:15:25 2016

@author: GrinevskiyAS
"""

from __future__ import division
import pandas as pd
import numpy as np
from numpy import sin,cos,tan,pi,sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib.gridspec as gridspec
from scipy import linalg as la
#from obspy.segy.segy import readSEGY


#поддержка кариллицы
font = {'family': 'Arial', 'weight': 'normal', 'size': 13}
font_annot = {'family': 'Arial', 'weight': 'normal', 'size': 10}
mpl.rc('font', **font)


def lin_regr(welldata,attr):
#    welldata: Nwells x 1
#    attr: Nwells x Nattr
    nwells = len(welldata)
    if (attr.ndim == 1):
        nattr = 1
    else:
        nattr = attr.shape[1]

    #строим матрицу М и матрицу наблюденных амплитуд R
    M1 = np.ones((nwells,1))
    M2 = attr.reshape(nwells, nattr)
    M=np.hstack((M1,M2))
    
    #методом МНК определяем параметры аппроксимации
    lam=1e-7
    res=la.inv(M.T.dot(M)+lam*np.eye(nattr+1)).dot(M.T).dot(welldata)
        
    #считаем аппроксимирующую кривую и погрешность
    appr=np.dot(M,res)
    err=sqrt((1/nwells)*np.sum(np.square( appr-welldata.ravel() )))
    kdet = np.sum((appr-np.mean(appr))**2)/np.sum((welldata-np.mean(welldata))**2)
    cc = np.sum((appr-np.mean(appr))*(welldata-np.mean(welldata)))/(np.sqrt(np.sum((appr-np.mean(appr))**2))*np.sqrt(np.sum((welldata-np.mean(welldata))**2)))

    return res, err, appr, kdet, cc
    

data_wells = pd.read_csv(r"D:\Projects\Komandirshor\HRS\attr_an_d2_data\D2kl\ForLinReg_D2kl_Kpor_well_no1.txt",
                         delim_whitespace=True, index_col=False)
welldata = data_wells.iloc[:,1].values

data_attr = pd.read_csv(r"D:\Projects\Komandirshor\HRS\attr_an_d2_data\D2kl\ForLinReg_D2kl_attr_no1.txt",
                         delim_whitespace=True, index_col=False)
attr_all = data_attr.iloc[:,1:].values
attr_names = data_attr.axes[1][1:]

Nattr=data_attr.shape[1]-1
Nwells=len(welldata)

well_names = data_wells.iloc[:,0].values.astype(str)

res_list = [None]*Nattr
err_list = [None]*Nattr
appr_list = [None]*Nattr
kdet_list = [None]*Nattr

kdet_mat = np.zeros((Nattr,Nattr))
cc_mat = np.zeros((Nattr,Nattr))

for i, attrname in enumerate(attr_names):
    for j, attrname2 in enumerate(attr_names):
        if (not (i==j) ):
            resij, errij, apprij, kdetij, ccij = lin_regr(welldata, np.column_stack((attr_all[:,i],attr_all[:,j])))
            kdet_mat[i,j] = kdetij
            cc_mat[i,j] = ccij
        else:
            resi, erri, appri, kdeti, cci = lin_regr(welldata, attr_all[:,i])
            cc_mat[i,i] = cci
            kdet_mat[i,i] = kdeti


ind_attr_1 = 6
ind_attr_2 = 9
attr1 = attr_all[:, ind_attr_1]
attr2 = attr_all[:, ind_attr_2]
attr_both = np.column_stack((attr1, attr2))
res1, err1, appr1, kdet1, cc1 = lin_regr(welldata, attr1)
res2, err2, appr2, kdet2, cc1 = lin_regr(welldata, attr2)
res_both, err_both, appr_both, kdet_both, cc_both = lin_regr(welldata, attr_both)

fgr = plt.figure(facecolor = 'w', figsize = [15,5])
ax1=fgr.add_subplot(1, 3, 1)
ax1.scatter(attr1, welldata)
ax1.set_title(attr_names[ind_attr_1])
ax2=fgr.add_subplot(132)
ax2.scatter(attr2, welldata)
ax2.set_title(attr_names[ind_attr_2])
ax_all=fgr.add_subplot(133)
ax_all.set_title('{0:.3f} + {1:.3f}*[{3}] + {2:.5f}*[{4}]'.format(res_both[0], res_both[1], res_both[2], attr_names[ind_attr_1], attr_names[ind_attr_2]))

ax1.plot(attr1, appr1, 'r')
ax2.plot(attr2, appr2, 'r')

attr_compl = np.dot(np.hstack((np.ones((np.size(welldata),1)), attr_both)), res_both)
ax_all.scatter(attr_compl, welldata)
ax_all.plot(attr_compl, appr_both, 'r')

for ind, wellname in enumerate(well_names):
    ax1.text(attr1[ind]+0.1, welldata[ind]+0.1, wellname, fontdict=font_annot)    
    ax2.text(attr2[ind]+0.1, welldata[ind]+0.1, wellname, fontdict=font_annot)    
    ax_all.text(attr_compl[ind]+0.1, welldata[ind]+0.1, wellname, fontdict=font_annot)

print '[WELL] = {0:.3f} + {1:.3f}*[{4}]. error = {2:.3f}. R2 = {3:.3f}'.format(res1[0],res1[1], err1, kdet1, attr_names[ind_attr_1])
print '[WELL] = {0:.3f} + {1:.3f}*[{4}]. error = {2:.3f}. R2 = {3:.3f}'.format(res2[0],res2[1], err2, kdet2,attr_names[ind_attr_2])
print '[WELL] = {0:.3f} + {1:.3f}*[{5}] + {2:.5f}*[{6}]. error = {3:.3f}. R2 = {4:.3f}'.format(res_both[0], res_both[1], res_both[2], err_both, kdet_both,attr_names[ind_attr_1],attr_names[ind_attr_2])


for ax in [ax1,ax2,ax_all]:
    ax.grid(True)
    ax_all.set_xlim(ax_all.get_ylim())