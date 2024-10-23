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
from models.IV_Model import Train_IVMLP
from models.MLP import Train_MLP
from keras.models import load_model
from utils.processing_Synthetic import process_Synthetic
from utils.tools import process_mean,process_input
from utils.tools import process_Normalize,model_test,cal_change
from sklearn.preprocessing import StandardScaler
from algorithms.Shap import cc_shap
from utils.plot import plot_boxs
from algorithms.intergrated_gradients import calculate_mlp_integrated_gradients,calculate_iv_integrated_gradients
import warnings
import random
warnings.filterwarnings('ignore')
import pickle
from sklearn.utils import shuffle

type = 'B'
rho = 1.0
file_name = "type{}_rho_{}".format(type,rho)
ratios = [0.125,0.25,0.375,0.5]


x,t,y,z = process_Synthetic(type,rho)
#x_scaled, t_scaled, z_scaled = process_Normalize(x,t,z,True)
#x_train, x_val, t_train, t_val, y_train, y_val, z_train, z_val = train_test_split(x_scaled, t_scaled, y, z_scaled, test_size=0.1, random_state=42)
x_train, x_val, t_train, t_val, y_train, y_val, z_train, z_val = train_test_split(x, t, y, z, test_size=0.1, random_state=42)
mean_data = process_mean(t_train, x_train)
input_shape = (len(x_train[0])+1,)

model_name = "Synthetic_IV_{}_{}".format(type,rho)
Train_IVMLP(x_train, t_train, y_train, z_train, input_shape, model_name)
model = load_model('./model_parameters/{}.h5'.format(model_name))
model.compile(optimizer='adam', loss='mse')
input_data = [t_train,x_train]
pic_name = model_name
model_test(model,input_data,y_train,pic_name)

model_name = "Synthetic_MLP_{}_{}".format(type,rho)
Train_MLP(x_train, t_train , y_train, input_shape, model_name)
model = load_model('./model_parameters/{}.h5'.format(model_name))
model.compile(optimizer='adam', loss='mse')
input_data = np.column_stack([t_train,x_train]) 
pic_name = model_name
model_test(model,input_data,y_train,pic_name)

all_data = []
for r in range(len(ratios)):
    err_DeepIV = []
    err_MLP = []
    err_IG_DeepIV = []
    err_IG_MLP = []

    for idx in range(1000):
        input_data = process_input(t_train,x_train,idx)
        base = input_data.copy()
        base[0] -= ratios[r]
        base[2] -= ratios[r]*6

        model_name = "Synthetic_IV_{}_{}".format(type,rho)
        SV_DeepIV = cc_shap(model_name, input_data, base, True, 100)
        IG_DeepIV = calculate_iv_integrated_gradients(model_name,input_data,base)
    
        model_name = "Synthetic_MLP_{}_{}".format(type,rho)
        SV_MLP = cc_shap(model_name, input_data, base,False, 100)
        IG_MLP = calculate_mlp_integrated_gradients(model_name,input_data,base)

        real_change = 0
        if type == 'A':
            real_change = ratios[r]
        else:
            real_change = cal_change(input_data,ratios[r])
    
        err_DeepIV.append(abs(SV_DeepIV[0]-real_change))
        err_MLP.append(abs(SV_MLP[0]-real_change))
        err_IG_DeepIV.append(abs(IG_DeepIV[0][0]-real_change))
        err_IG_MLP.append(abs(IG_MLP[0][0]-real_change))


    all_data.append(err_DeepIV)
    all_data.append(err_MLP)
    all_data.append(err_IG_DeepIV)
    all_data.append(err_IG_MLP)

np.savetxt('./results/{}.txt'.format(file_name), np.array(all_data), fmt='%.3f')
read_data = np.loadtxt('./results/{}.txt'.format(file_name), dtype=float)
all_data = [read_data[i, :] for i in range(read_data.shape[0])]
for i in range(len(all_data)):
    print(all_data[i].mean())
plot_boxs(all_data,file_name,rho)


