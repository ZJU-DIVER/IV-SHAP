import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.callbacks import EarlyStopping
from models.MixtureDensityNet import MixtureDensityNetwork
from utils.tools import process_mean,process_input
from utils.tools import process_Normalize,model_test,cal_change
from utils.processing_Synthetic import process_Synthetic
from utils.plot import plot_boxs_XGBRegressor
from algorithms.IV_SHAP import cc_shap_XGBRegressor
import warnings
import copy
from xgboost import XGBRegressor
import random
warnings.filterwarnings('ignore')
import pickle
from sklearn.utils import shuffle

type = 'B'
rho = 0.2
file_name = "XGBoost_type{}_rho{}".format(type,rho)
ratios = [0.125,0.25,0.375,0.5]


x,t,y,z = process_Synthetic(type,rho)
x_train, x_val, t_train, t_val, y_train, y_val, z_train, z_val = train_test_split(x, t, y, z, test_size=0.1, random_state=42)
mean_data = process_mean(t_train, x_train)
input_shape = (len(x_train[0])+1,)

num_components = 1
input_dim = 7
num_continuous_features = 7  
num_discrete_features = 0
category_counts = [0, 0] 
embedding_dim = 4
num_samples = 5000

mdn = MixtureDensityNetwork(num_components, input_dim, num_discrete_features, category_counts, embedding_dim)
mdn.compile(0.001)
mdn.fit(np.column_stack([z_train,x_train]), t_train)
t_pred = mdn.predict(np.column_stack([z_train,x_train]))
t_samples = mdn.sample_from_mdn_output(t_pred, num_samples=20)

t_iv = copy.copy(t_train)
for i in range(len(t_iv)):
    t_iv[i] = 0
for i in range(len(t_iv)):
    for j in range(20):
        t_iv[i] += t_samples[j][i]
    t_iv[i] /= 20
model_IV = XGBRegressor()
model_IV.fit(np.column_stack([t_iv,x_train]),y_train)

model_base = XGBRegressor()
model_base.fit(np.column_stack([t_train,x_train]),y_train)

all_data = []
for r in range(len(ratios)):
    err_DeepIV = []
    err_MLP = []

    for idx in range(100):
        input_data = process_input(t_train,x_train,idx)
        base = input_data.copy()
        base[0] -= ratios[r]
        base[2] -= ratios[r]*6

        SV_DeepIV = cc_shap_XGBRegressor(model_IV, input_data, base, False, 100)
        SV_MLP = cc_shap_XGBRegressor(model_base, input_data, base,False, 100)

        real_change = 0
        if type == 'A':
            real_change = ratios[r]
        else:
            real_change = cal_change(input_data,ratios[r])
    
        err_DeepIV.append(abs(SV_DeepIV[0]-real_change))
        err_MLP.append(abs(SV_MLP[0]-real_change))

    all_data.append(err_DeepIV)
    all_data.append(err_MLP)

np.savetxt('./results/{}.txt'.format(file_name), np.array(all_data), fmt='%.3f')
read_data = np.loadtxt('./results/{}.txt'.format(file_name), dtype=float)
all_data = [read_data[i, :] for i in range(read_data.shape[0])]
for i in range(len(all_data)):
    print(all_data[i].mean())
plot_boxs_XGBRegressor(all_data,file_name,rho)







