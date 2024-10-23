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
from utils.processing_angrist import process_Angrist,Sample_Balanced_data
from utils.processing_Synthetic import process_complex_Synthetic,process_Synthetic,process_high_dimensional_Synthetic
from utils.tools import process_Normalize
from algorithms.Shap import cc_shap
from algorithms.cc_neyman import ccshap_neyman
from algorithms.cc_confidence import ccshap_confidence
import warnings
import random
warnings.filterwarnings('ignore')
import pickle
from sklearn.utils import shuffle


x,t,y,z = process_high_dimensional_Synthetic()
x_scaled, t_scaled, z_scaled = process_Normalize(x,t,z,True)
# Split the data into training and validation sets
x_train, x_val, t_train, t_val, y_train, y_val, z_train, z_val = train_test_split(x_scaled, t_scaled, y, z_scaled, test_size=0.1, random_state=42)
mean_data = process_mean(t_train, x_train)
input_shape = (len(x_train[0])+1,)

cc_lists = []
neyman_lists = []
confidence_lists = []
base_lists = []
cc_err = 0
neyman_err = 0
confidence_err = 0
for idx in range(1000):
    input = process_input(t_train,x_train,idx)
    model_name = "Synthetic_high_dimension_MLP"
    SV_cc = cc_shap(model_name, input, mean_data, False, 10000)
    SV_neyman = ccshap_neyman(model_name, input, mean_data, False, 10, 10000)
    SV_confidence = ccshap_confidence(model_name, input, mean_data, False, 10, 10000)
    SV_base = cc_shap(model_name, input, mean_data, False, 100000)

    print('finish ',idx)

    for i in range(len(SV_cc)):
        cc_lists.append(SV_cc[i])
    for i in range(len(SV_base)):
        base_lists.append(SV_base[i])
    for i in range(len(SV_neyman)):
        neyman_lists.append(SV_neyman[i])
    for i in range(len(SV_confidence)):
        confidence_lists.append(SV_confidence[i])

    cc_err += np.sum(np.abs(SV_cc-SV_base))
    neyman_err += np.sum(np.abs(SV_neyman-SV_base))
    confidence_err += np.sum(np.abs(SV_confidence-SV_base))

    print("error of cc: ", cc_err)
    print("error of neyman: ", neyman_err)
    print("error of confidence: ", confidence_err)

print('initilization 10, samples 10000')

sorted_indices = np.argsort(base_lists)
base_lists = np.array(base_lists)
cc_lists = np.array(cc_lists)
neyman_lists = np.array(neyman_lists)
confidence_lists = np.array(confidence_lists)
sorted_base = base_lists[sorted_indices]
sorted_a = cc_lists[sorted_indices]
sorted_b = neyman_lists[sorted_indices]
sorted_c = confidence_lists[sorted_indices]


plt.figure(figsize=(12, 6))
plt.plot(range(len(sorted_base)), sorted_base, label='base', color='g')
plt.plot(range(len(sorted_a)), sorted_a, label='cc', color='b')
plt.plot(range(len(sorted_b)), sorted_b, label='neyman', color='r')
plt.plot(range(len(sorted_c)), sorted_c, label='confidence', color='y')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Comparison of SV approximation')
plt.legend()
plt.savefig('./pictures/{}.png'.format("All"))

    

































