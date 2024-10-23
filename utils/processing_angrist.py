import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from algorithms.Shap import cc_shap
import random
from sklearn.utils import shuffle

def process_Angrist(selected_columns = ['AGE', 'EDUC', 'LWKLYWGE', 'QOB','ENOCENT','ESOCENT','MARRIED','RACE']):
    csv_path = "./data/NEW7080.csv"
    # Read the CSV file into a DataFrame
    data_df = pd.read_csv(csv_path, delimiter=',')
    # Split the single column into multiple columns
    # Remove the long string with comma-separated column names
    data_df = data_df[['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27']]

    # Rename the columns based on the description file or prior knowledge
    column_names = {
        'v1': 'AGE',
        'v2': 'AGEQ',
        'v4': 'EDUC',
        'v5': 'ENOCENT',
        'v6': 'ESOCENT',
        'v9': 'LWKLYWGE',
        'v10': 'MARRIED',
        'v11': 'MIDATL',
        'v12': 'MT',
        'v13': 'NEWENG',
        'v16': 'CENSUS',
        'v18': 'QOB',
        'v19': 'RACE',
        'v20': 'SMSA',
        'v21': 'SOATL',
        'v24': 'WNOCENT',
        'v25': 'WSOCENT',
        'v27': 'YOB'
        # ... and so on for other columns as needed
    }
    data_df.rename(columns=column_names, inplace=True)
    # Select relevant columns
    #selected_columns = ['AGE', 'EDUC', 'LWKLYWGE', 'QOB','ENOCENT','ESOCENT','MARRIED','RACE']
    data_df = data_df[(data_df['YOB'] >= 1920) & (data_df['YOB'] <= 1929)]
    data_df = data_df[selected_columns]

    # Prepare data for training
    TYZ = ['EDUC','LWKLYWGE','QOB']
    #X = data_df[['AGE','ENOCENT','ESOCENT','MARRIED','RACE']]
    X = data_df[[x for x in selected_columns if x not in TYZ]]
    t = data_df['EDUC']
    y = data_df['LWKLYWGE']
    z = data_df['QOB']  # Assuming AGEQ represents the birth quarter   

    return X,t,y,z 

def Sample_Balanced_data():
    # Load the dataset
    data_df = pd.read_csv("./data/SAMPLED.csv")
    # Remove the long string with comma-separated column names
    data_df = data_df[['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27']]
    data_race_1 = data_df[data_df['v19'] == 1]
    num_race_1 = len(data_race_1)
    data_race_0_sampled = data_df[data_df['v19'] == 0].sample(n=num_race_1, random_state=42)
    new_data = pd.concat([data_race_1, data_race_0_sampled])
    new_data = shuffle(new_data, random_state=42)
    new_data.to_csv("./data/SAMPLED_BALANCED.csv", index=False)

def sample_csv(input_csv_path, output_csv_path, sample_size=100000, random_state=None):
    df = pd.read_csv(input_csv_path)
    sampled_df = df.sample(n=sample_size, random_state=random_state)
    sampled_df.to_csv(output_csv_path, index=False)

def Sample_IV():
    df = pd.read_csv("./data/SAMPLED_BALANCED.csv")
    final_df = pd.DataFrame()

    conditions = [
        ('v18', 1, 0, 0.1),
        ('v18', 2, 0.4, 0.5),
        ('v18', 3, 0.7, 0.8),
        ('v18', 4, 0.9, 1)
    ]

    for column, value, lower_percentile, upper_percentile in conditions:
        filtered_df = df[df[column] == value]
    
        # Calculate the lower and upper bounds for 'v4'
        lower_bound = filtered_df['v4'].quantile(lower_percentile)
        upper_bound = filtered_df['v4'].quantile(upper_percentile)

        print(lower_bound,upper_bound)
        # Further filter the DataFrame based on the percentile range for 'v4'
        percentile_filtered_df = filtered_df[(filtered_df['v4'] >= lower_bound) & (filtered_df['v4'] <= upper_bound)]

        # Randomly sample 500 records (or fewer if there are not enough)
        sampled_df = percentile_filtered_df.sample(min(len(percentile_filtered_df), 1000))
    
        # Append the sampled data to the final DataFrame
        final_df = pd.concat([final_df, sampled_df])

    final_df.to_csv("./data/SAMPLED_IV.csv", index=False)