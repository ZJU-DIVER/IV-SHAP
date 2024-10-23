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
from utils.tools import process_Angrist,process_mean,process_input,process_Synthetic,process_Normalize,plot_compare_sorted_models
from algorithms.Shap import cc_shap
import warnings
warnings.filterwarnings('ignore')
import pickle

x,t,y,z = process_Synthetic()


x_scaled, t_scaled, z_scaled = process_Normalize(x,t,z)
# Split the data into training and validation sets
x_train, x_val, t_train, t_val, y_train, y_val, z_train, z_val = train_test_split(x_scaled, t_scaled, y, z_scaled, test_size=0.1, random_state=42)

input_shape = (len(x_train[0])+1,)
model_name = "Synthetic_IVMLP"
Train_IVMLP(x_train, t_train, y_train, z_train, input_shape, model_name)
model_name = "Synthetic_MLP"
Train_MLP(x_train, t_train , y_train, input_shape, model_name)

model_name = "Synthetic_IVMLP"
model = load_model('./model_parameters/{}.h5'.format(model_name))
model.compile(optimizer='adam', loss='mse')
y_pred_IVMLP = model.predict([t_train,x_train])

model_name = "Synthetic_MLP"
model = load_model('./model_parameters/{}.h5'.format(model_name))
model.compile(optimizer='adam', loss='mse')
y_pred_MLP = model.predict(np.column_stack([t_train,x_train]))


mean_data = process_mean(t_train, x_train)
err_IVMLP = 0
err_MLP = 0
for idx in range(1000):
    input_data = process_input(t_train,x_train,idx)

    model_name = "Synthetic_IVMLP"
    SV_IVMLP = cc_shap(model_name, input_data, mean_data, True, 100, 4)
    
    model_name = "Synthetic_MLP"
    SV_MLP = cc_shap(model_name, input_data, mean_data,False, 100, 4)
    
    mean_ori_data = process_mean(t, x)
    input_ori_data = process_input(t,x, idx)

    base = input_ori_data[0] - mean_ori_data[0]
    err_IVMLP += np.fabs(SV_IVMLP[0]/base)
    err_MLP += np.fabs(SV_MLP[0]/base)
    print('Error of IV-Model: ', err_IVMLP)
    print('Error of MLP: ', err_MLP)








