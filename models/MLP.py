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

def Train_MLP(X_train, t_train , y_train, input_shape, model_name):

    # Model Structure
    regression_model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=input_shape),  # 直接在这里指定输入维度
        keras.layers.Dropout(0.17),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.17),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.17),
        keras.layers.Dense(1)
    ])

    # Compile
    regression_model.compile(optimizer='adam', loss='mean_squared_error')

    # Train
    mlp_model = regression_model.fit(np.column_stack([t_train,X_train]), y_train, epochs=100,
                               batch_size=32,callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    
    save_model(regression_model,'./model_parameters/{}.h5'.format(model_name))