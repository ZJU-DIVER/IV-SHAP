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
from utils.tools import process_mean,process_input
from utils.processing_griliches import process_griliches
from utils.tools import process_Normalize,plot_compare_sorted_models,model_test,find_similar_data,find_similar_hidden_data
from algorithms.Shap import cc_shap
from algorithms.intergrated_gradients import calculate_mlp_integrated_gradients,calculate_iv_integrated_gradients
import warnings
import random
warnings.filterwarnings('ignore')
import pickle
from sklearn.utils import shuffle
import copy

#x,t,y,z = process_griliches(['s','age','expr','tenure','med','mrt','rns','smsa','lw','iq','kww'])
x,t,y,z = process_griliches(['s','age','expr','tenure','med','mrt','rns','smsa','lw'])
x_scaled, t_scaled, z_scaled = process_Normalize(x,t,z,False)
# Split the data into training and validation sets
x_train, x_val, t_train, t_val, y_train, y_val, z_train, z_val = train_test_split(x_scaled, t_scaled, y, z_scaled, test_size=0.1, random_state=42)
mean_data = process_mean(t_train, x_train)

# input_shape = (len(x_train[0])+1,)

x_all,t_all,y_all,z_all = process_griliches(['s','age','expr','tenure','med','mrt','rns','smsa','lw','iq','kww'])
x_scaled_all, t_scaled_all, z_scaled_all = process_Normalize(x_all,t_all,z_all,False)
# Split the data into training and validation sets
x_train_all, x_val_all, t_train_all, t_val_all, y_train_all, y_val_all, z_train_all, z_val_all = train_test_split(x_scaled_all, t_scaled_all, y_all, z_scaled_all, test_size=0.1, random_state=42)
mean_data_all = process_mean(t_train_all, x_train_all)

model_name = "Griliches_IV_All"
Train_IVMLP(x_train, t_train, y_train, z_train, input_shape, model_name)
#Train_MLP(x_train, t_train , y_train, input_shape, model_name)
model = load_model('./model_parameters/{}.h5'.format(model_name))
model.compile(optimizer='adam', loss='mse')
input_data = [t_train,x_train]
#input_data = np.column_stack([t_train,x_train])
pic_name = model_name
model_test(model,input_data,y_train,pic_name)

err_DeepIV = 0
err_MLP = 0
err_DeepIV_IG = 0
err_MLP_IG = 0
err_DeepIV_All = 0
err_MLP_IG_All = 0

s = np.unique(t_scaled_all)

IV_list = []
MLP_list = []
Dif_list = []

count = 0
for idx in range(len(x_train_all)):
    input = process_input(t_train,x_train,idx)
    base = input.copy()
    smaller_elements = [elem for elem in s if elem < base[0]]
    sorted_smaller_elements = sorted(smaller_elements,reverse=True)
    if len(sorted_smaller_elements) >= 5:
        base[0] = sorted_smaller_elements[4]
    else:
        continue
    model_name = "Griliches_IV"
    SV_DeepIV = cc_shap(model_name, input, base, True, 100)
    IG_IV = calculate_iv_integrated_gradients(model_name,input,base)

    model_name = "Griliches_MLP"
    SV_MLP = cc_shap(model_name, input, base,False, 100)
    IG_MLP = calculate_mlp_integrated_gradients(model_name,input,base)

    input_all = process_input(t_train_all,x_train_all,idx)
    base = input_all.copy()
    smaller_elements = [elem for elem in s if elem < base[0]]
    sorted_smaller_elements = sorted(smaller_elements,reverse=True)
    if len(sorted_smaller_elements) >= 5:
        base[0] = sorted_smaller_elements[4]
    else:
        continue

    model_name = "Griliches_MLP_All"
    SV_MLP_All = cc_shap(model_name, input_all, base, False, 100)
    IG_MLP_All = calculate_mlp_integrated_gradients(model_name,input_all,base)

    count += 1
   
    err_DeepIV += np.abs((SV_DeepIV[0]/y_train.iloc[idx]))
    err_MLP += np.abs((SV_MLP[0])/y_train.iloc[idx])
    err_DeepIV_IG += np.abs((IG_IV[0][0])/y_train.iloc[idx])
    err_MLP_IG += np.abs((IG_MLP[0][0])/y_train.iloc[idx])
    err_DeepIV_All += np.abs(SV_MLP_All[0]/y_train.iloc[idx])
    err_MLP_IG_All += np.abs(IG_MLP_All[0][0]/y_train.iloc[idx])
    
    print('error of IV: ', np.abs((err_DeepIV-err_DeepIV_All)/err_DeepIV_All))
    print('error of MLP: ', np.abs((err_MLP-err_DeepIV_All)/err_DeepIV_All))
    print('error of IV-IG: ', np.abs((err_DeepIV_IG-err_MLP_IG_All)/err_MLP_IG_All))
    print('error of MLP-IG: ', np.abs((err_MLP_IG-err_MLP_IG_All)/err_MLP_IG_All))
    

    





































