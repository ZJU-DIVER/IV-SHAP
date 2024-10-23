import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.models import save_model

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def Train_MLP_Classifier(X_train, t_train, y_train, input_shape, model_name):
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=2)

    # Define the neural network model
    classification_model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        keras.layers.Dropout(0.17),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.17),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.17),
        keras.layers.Dense(2, activation='softmax')  # Modify the output layer
    ])

    adam_optimizer = Adam(learning_rate=0.001)

    # Compile the model, use categorical_crossentropy as the loss function
    classification_model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    classification_model.fit(np.column_stack([t_train, X_train]), y_train, epochs=50,
                             batch_size=32, callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    
    classification_model.save('./model_parameters/{}.h5'.format(model_name))
