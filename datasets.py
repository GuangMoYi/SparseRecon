import numpy as np
import h5py
import pickle
from scipy.stats import zscore
from scipy.ndimage import convolve
from sklearn.preprocessing import RobustScaler
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import mode
num1 = 0
num2 = 2
def NOAA():
    # 我的数据是509*60*52*2的，但是由于数据的排列是中的V，是改变X，Y不变的排列，所以代码要将数据reshape为52*60
    f = h5py.File('Data/pivdata11.mat', 'r')
    # f = h5py.File('Data/5093122pivdata11.mat', 'r')
    sst = np.nan_to_num(np.array(f['pivdata']))
    shape1, shape2 = 52, 60
    num_frames, variables = sst.shape
    sea = np.zeros((num_frames, shape1, shape2, 2))
    for t in range(num_frames):
        reshaped_data = sst[t, :].reshape(shape1, shape2, 2, order='C')
        sea[t] = reshaped_data
    # sea = sea / 2
    sea_change = sea
    sea = (sea_change - num1) / num2
    return sea


def pipe():
    # 使用这个最简单的程序读取二进制数据吧
    with open("Data/Turbulent/ch_2Dxysec.pickle", 'rb') as f:
        pipe = pickle.load(f)
        pipe /= np.abs(pipe).max()
    return pipe


def cylinder():
    with open('Data/Cylinder/Cy_Taira.pickle', 'rb') as f:
        cyl = pickle.load(f) / 11.0960
    return cyl


def plume():
    with h5py.File('Data/Plume/concentration.h5', "r") as f:
        plume_3D = f['cs']
        plume_3D = np.array(plume_3D)
        plume_3D /= plume_3D.max()
    return plume_3D


def porous():
    with h5py.File('Data/Pore/rho_1.h5', "r") as f:
        pore = f['rho'][:]
    return pore


def isotropic3D():
    with h5py.File('Isotropic/scalarHIT_fields100.h5', "r") as f:
        return np.array(f['fields'])
