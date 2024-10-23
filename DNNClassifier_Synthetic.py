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
from keras.models import load_model
from models.MixtureDensityNet import MixtureDensityNetwork
from models.DNNClassifier import Train_MLP_Classifier
from utils.tools import process_mean,process_input
from utils.tools import process_Normalize,model_test,cal_change
from utils.processing_Synthetic import process_Synthetic_Classifier,cal_equal_change
from utils.plot import plot_boxs
import copy
from algorithms.IV_SHAP import cc_shap_DNNClassifier
from algorithms.IG import calculate_mlp_integrated_gradients
import warnings
import xgboost as xgb
import random
warnings.filterwarnings('ignore')
import pickle
from sklearn.utils import shuffle

type = 'A'
rho = 0.2
file_name = "DNNClassifier_type{}_rho{}".format(type,rho)
ratios = [0.125,0.25,0.375,0.5]

x,t,y,z = process_Synthetic_Classifier(type,rho)
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
mdn.compile(0.0001)
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


model_IV = "DNNClassifierIV_{}_{}".format(type,rho)
Train_MLP(x_train, t_iv , y_train, input_shape, model_IV)

model_base = "DNNClassifierMLP_{}_{}".format(type,rho)
Train_MLP(x_train, t_train , y_train, input_shape, model_base)

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
        base[2] -= cal_equal_change(input_data,ratios[r],type)

        IG_IV = calculate_mlp_integrated_gradients(model_IV,input_data,base)
        IG_MLP = calculate_mlp_integrated_gradients(model_base,input_data,base)
        SV_DeepIV = cc_shap_DNNClassifier(model_IV, input_data, base, False, 125)
        SV_MLP = cc_shap_DNNClassifier(model_base, input_data, base,False, 125)
        
        err_DeepIV.append(abs(SV_DeepIV[0]-SV_DeepIV[2]))
        err_MLP.append(abs(SV_MLP[0]-SV_MLP[2]))
        err_IG_DeepIV.append(abs(IG_IV[0][0]-IG_IV[0][2]))
        err_IG_MLP.append(abs(IG_MLP[0][0]-IG_MLP[0][2]))

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







