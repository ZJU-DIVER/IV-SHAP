from econml.iv.nnet import DeepIV
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.models import save_model
import pickle

def Train_IVMLP(X_train, t_train,  y_train, z_train, input_shape, model_name):
    # Define the treatment model
    treatment_model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        keras.layers.Dropout(0.17),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.17),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.17)
    ])

    # Define the response model
    response_model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        keras.layers.Dropout(0.17),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.17),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.17),
        keras.layers.Dense(1)
    ])

    # Initialize the IVMLP model
    deepIvEst = DeepIV(n_components=5,
                   m=lambda z, x: treatment_model(keras.layers.concatenate([z, x])),
                   h=lambda t, x: response_model(keras.layers.concatenate([t, x])),
                   n_samples=5,
                   use_upper_bound_loss=False,
                   n_gradient_samples=1,
                   optimizer=keras.optimizers.Adam(learning_rate=0.001),
                   first_stage_options={"epochs": 100, "batch_size": 32},
                   second_stage_options={"epochs": 100, "batch_size": 32})

    # Fit the DeepIV model
    deepIvEst.fit(Y=y_train, T=t_train, X=X_train, Z=z_train)

    save_model(deepIvEst._effect_model, './model_parameters/{}.h5'.format(model_name))