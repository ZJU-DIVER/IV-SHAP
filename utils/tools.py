import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def preprocess_adult():
    # Read the dataset
    data = pd.read_csv('./data_sets/adult.data', header=None)

    # Naming the feature columns
    columns = ['Age', 'Workclass', 'Final Weight', 'Education', 'Education Number', 'Marital Status',
               'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss',
               'Hours per Week', 'Native Country', 'Income']

    # Replace missing values with NaN
    data = data.replace('?', np.nan)

    # Delete rows with missing values
    data = data.dropna()

    # Reset column names
    data.columns = columns

    # Select features without missing values
    features = data.drop(['Workclass', 'Final Weight', 'Education', 'Education Number', 'Marital Status', 'Occupation', 'Relationship',
                          'Race', 'Sex', 'Native Country', 'Income'], axis=1)

    # Splitting features and labels
    X = features.to_numpy()
    Y = data['Income'].to_numpy()

    # Splitting into training and validation sets
    X_train = X[:600]
    Y_train = Y[:600]
    X_test = X[600:800]
    Y_test = Y[600:800]

    return X_train, Y_train, X_test, Y_test

def process_Angrist():
    csv_path = "./data/Part_NEW.csv"
    # Read the CSV file into a DataFrame
    data_df = pd.read_csv(csv_path, delimiter=',')
    # Split the single column into multiple columns
    # Remove the long string with comma-separated column names
    data_df = data_df[['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27']]

    # Rename columns based on description file or prior knowledge
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
    selected_columns = ['AGE', 'EDUC', 'LWKLYWGE', 'QOB', 'ENOCENT', 'ESOCENT', 'MARRIED', 'RACE']
    data_df = data_df[selected_columns]

    # Prepare data for training
    X = data_df[['AGE', 'ENOCENT', 'ESOCENT', 'MARRIED', 'RACE']]
    t = data_df['EDUC']
    y = data_df['LWKLYWGE']
    z = data_df['QOB']  # Assuming AGEQ represents the birth quarter   

    return X, t, y, z 

def process_input(t_train, X_train, idx):
    input_data = []
    input_data.append(t_train[idx])
    for i in range(len(X_train[idx])):
        input_data.append(X_train[idx][i])
    return input_data

def process_Synthetic():
    n = 5000
    e = np.random.uniform(low=0.0, high=5.0, size=(n,))
    X = np.random.uniform(low=0.0, high=5.0, size=(n, 3))
    z = np.random.uniform(low=0.0, high=5.0, size=(n,))

    # Initialize treatment variable
    t = np.sqrt(e * z) + e + z
    # Outcome equation 
    y = t + e*e + (X[:,0]**2 + X[:,1] + np.sqrt(X[:,2]))/10

    return X, t, y, z

def process_mean(t_train, x_train):
    # Convert t_train and x_train to NumPy arrays for easier manipulation
    t_train = np.array(t_train)
    x_train = np.array(x_train)

    # Calculate the mean of each column in t_train and x_train
    mean_data_t = np.mean(t_train, axis=0)
    mean_data_x = np.mean(x_train, axis=0)
    mean_data_t = mean_data_t.reshape(-1)
    # Combine the two lists of means into one
    mean_data = np.concatenate((mean_data_t, mean_data_x))
    return mean_data.tolist()

def generate_shap_predictive_data(input_data, mean_data, idxs, input_seperate):
    if input_seperate == False:
        # Convert the selected features into the format for MLP input
        shap_data = []
        for i in range(len(input_data)):
            if i in idxs:
                shap_data.append(input_data[i])
            else:
                shap_data.append(mean_data[i])
        return np.array(shap_data).reshape(1, -1)
    else:
        # Convert the selected features into the format for DeepIV input
        shap_t, shap_x = [], []
        for i in range(len(input_data)):
            if i in idxs:
                if i == 0:
                    shap_t.append(input_data[i])
                else:
                    shap_x.append(input_data[i])
            else:
                if i == 0:
                    shap_t.append(mean_data[i])
                else:
                    shap_x.append(mean_data[i])
        return [shap_t, [shap_x]]
    
def process_Normalize(x, t, z):
    # Normalize input data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    t_scaled_2D = scaler.fit_transform(t.reshape(-1, 1))
    t_scaled = t_scaled_2D.reshape(-1)
    z_scaled_2D = scaler.fit_transform(z.reshape(-1, 1))
    z_scaled = z_scaled_2D.reshape(-1)

    return x_scaled, t_scaled, z_scaled
    

def model_test(model, input_data, y_val, pic_name):
    # Using the trained model to make predictions on the validation dataset
    y_val_pred = model.predict(input_data).flatten()

    # Plotting the true values against the predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_val_pred, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values on Validation Dataset')
    plt.grid(True)
    plt.legend()
    plt.savefig("./pictures/{}.png".format(pic_name))

def plot_compare_sorted_models(y_train, y_deepiv, y_mlp, pic_name):
    sorted_indices = np.argsort(y_train)
    sorted_a = y_train[sorted_indices]
    sorted_b = y_deepiv[sorted_indices]
    sorted_c = y_mlp[sorted_indices]

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(sorted_a)), sorted_a, label='y_true', color='b')
    plt.plot(range(len(sorted_b)), sorted_b, label='y_deepiv', color='r')
    plt.plot(range(len(sorted_c)), sorted_c, label='y_mlp', color='g')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Comparison of Sorted a, Corresponding b and c')
    plt.legend()
    plt.savefig('./pictures/{}.png'.format(pic_name))

def cal_change(data, ratios):
    AB = (np.exp(data[1]) + data[2] + np.sqrt(data[2]) + np.exp(data[4])/2 + data[5]/2 + np.sqrt(data[6])/2) / 6 * data[0]
    A = (np.exp(data[1]) + data[2] + np.sqrt(data[2]) + np.exp(data[4])/2 + data[5]/2 + np.sqrt(data[6])/2) / 6 * (data[0] - ratios)
    B = (np.exp(data[1]) + (data[2] - ratios * 6) + np.sqrt(data[2]) + np.exp(data[4])/2 + data[5]/2 + np.sqrt(data[6])/2) / 6 * (data[0])
    null = (np.exp(data[1]) + (data[2] - ratios * 6) + np.sqrt(data[2]) + np.exp(data[4])/2 + data[5]/2 + np.sqrt(data[6])/2) / 6 * (data[0] - ratios)
    return (AB - A + B - null) / 2
