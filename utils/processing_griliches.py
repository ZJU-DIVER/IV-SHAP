import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from algorithms.Shap import cc_shap
import random
from sklearn.utils import shuffle

#lw log of week salary
#s education years
#expr the year of woking
#tenure the year in current company
#med mother's education year
#mr married
#rns is live in south usa
#smsa is live in big city

def process_griliches(selected_columns = ['s','age','expr','tenure','med','mrt','rns','smsa','lw']):
    csv_path = "./data/griliches76.csv"
    # Read the CSV file into a DataFrame
    data_df = pd.read_csv(csv_path, delimiter=',')
    # Split the single column into multiple columns
    
    data_df = data_df[selected_columns]

    # Prepare data for training
    TYZ = ['s','lw','med']
    #X = data_df[['AGE','ENOCENT','ESOCENT','MARRIED','RACE']]
    X = data_df[[x for x in selected_columns if x not in TYZ]]
    t = data_df['s']
    y = data_df['lw']
    z = data_df['med']  # Assuming AGEQ represents the birth quarter   

    return X,t,y,z 