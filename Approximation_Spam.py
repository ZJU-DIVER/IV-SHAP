import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf

from econml.iv.nnet import DeepIV
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
from utils.tools import process_mean,process_input,model_test
from utils.processing_angrist import process_Angrist
from utils.processing_spambase import process_Spambase
from utils.processing_Synthetic import process_Synthetic
from utils.tools import process_Normalize
from algorithms.Shap import cc_shap,mc_shap
from algorithms.cc_neyman import ccshap_neyman
from algorithms.cc_confidence import ccshap_confidence
from algorithms.cc_ProbConfidence import ccshap_prob_confidence
import warnings
import random
warnings.filterwarnings('ignore')
import pickle
from sklearn.utils import shuffle


# x,t,y,z = process_Angrist(['AGE','AGEQ','EDUC', 'LWKLYWGE', 'QOB','ENOCENT','ESOCENT','MARRIED','MIDATL', 'MT',
#          'NEWENG','CENSUS','RACE','SMSA','SOATL','WNOCENT','WSOCENT','YOB'])
x,t,y,z = process_Spambase()
x_scaled, t_scaled, z_scaled = process_Normalize(x,t,z,False)
# Split the data into training and validation sets
x_train, x_val, t_train, t_val, y_train, y_val, z_train, z_val = train_test_split(x_scaled, t_scaled, y, z_scaled, test_size=0.1, random_state=42)
mean_data = process_mean(t_train, x_train)
model_name = "Spam_MLP_ALL"

input_shape = (len(x_train[0])+1,)
Train_MLP(x_train, t_train , y_train, input_shape, model_name)
model = load_model('./model_parameters/{}.h5'.format(model_name))
model.compile(optimizer='adam', loss='mse')
input_data = np.column_stack([t_train,x_train]) 
pic_name = model_name
model_test(model,input_data,y_train,pic_name)

cc_lists = []
neyman_lists = []
confidence_lists = []
base_lists = []
mc_lists = []
err_cc = np.zeros(5)
err_neyman = np.zeros(5)
err_confidence = np.zeros(5)
err_mc = np.zeros(5)
for idx in range(1000):
    
    input = process_input(t_train,x_train,idx)
    SV_base = cc_shap(model_name, input, mean_data, False, 56*1000)
    for i in range(5):
        SV_mc = mc_shap(model_name, input, mean_data, False, 56*(i+1)*100)
        SV_cc = cc_shap(model_name, input, mean_data, False, 56*(i+1)*100)
        SV_neyman = ccshap_neyman(model_name, input, mean_data, False,2, 56*(i+1)*100)
        SV_confidence = ccshap_prob_confidence(model_name, input, mean_data, False,2, 56*(i+1)*100)

        for j in range(len(SV_mc)):
            err_mc[i] += np.abs(SV_mc[j]-SV_base[j])
            err_cc[i] += np.abs(SV_cc[j]-SV_base[j])
            err_neyman[i] += np.abs(SV_neyman[j]-SV_base[j])
            err_confidence[i] += np.abs(SV_confidence[j]-SV_base[j])

    print('err mc',err_mc)
    print('err cc',err_cc)
    print('err neyman',err_neyman)
    print('err confidence',err_confidence)


    

































