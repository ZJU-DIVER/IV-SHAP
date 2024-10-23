import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from algorithms.Shap import cc_shap
import random
from sklearn.utils import shuffle

def process_Spambase():
    data_file_path = './data/spambase.data'  # Please replace with your file path

    # Path to save the CSV file
    csv_file_path = './data/spambase.csv'  # Please replace with the CSV file path you wish to save to

    # Column names: v1, v2, ..., v50, y
    column_names = [f'v{i}' for i in range(1, 56)] + ['y']

    # Read the .data file
    # Assuming the file is separated by spaces or tabs, adjust the delimiter as per your requirement
    data = pd.read_csv(data_file_path, header=None, names=column_names, sep=',', engine='python')

    # Save as a CSV file
    data.to_csv(csv_file_path, index=False)

    # Prepare data for training
    TYZ = ['v1','v2','y']
    #X = data_df[['AGE','ENOCENT','ESOCENT','MARRIED','RACE']]
    X = data[[x for x in column_names if x not in TYZ]]
    t = data['v1']
    y = data['y']
    z = data['v2']  # Assuming AGEQ represents the birth quarter   

    return X, t, y, z 
