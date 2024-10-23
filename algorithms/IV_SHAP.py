import sys,os
path = os.path.dirname("/home/..")
sys.path.append(path)

import datetime
import math
import random
from functools import partial
from multiprocessing import Pool
import numpy as np
from scipy.special import comb
from tqdm import trange
import copy
from keras.models import load_model
from utils.tools import generate_shap_predictive_data


def mc_shap(model_name, input_data, mean_data, input_seperate, m) -> np.ndarray:
    """Compute the Shapley value by sampling complementary contributions
    """

    n = len(input_data)
    local_state = np.random.RandomState(None)
    utility = np.zeros(n)
    count = np.zeros(n)
    idxs = np.arange(n)
    model = load_model('./model_parameters/{}.h5'.format(model_name))
    model.compile(optimizer='adam', loss='mse')
    for _ in trange(m):
        local_state.shuffle(idxs)
        j = random.randint(1, n)
        p_1 = model.predict(generate_shap_predictive_data(input_data,mean_data,idxs[:j],input_seperate))
        p_2 = model.predict(generate_shap_predictive_data(input_data,mean_data,idxs[:j-1],input_seperate))
        u_1, u_2 = p_1[0][0], p_2[0][0]
        utility[idxs[j-1]] += u_1 - u_2
        count[idxs[j-1]] += 1

    sv = np.zeros(n)
    for i in range(n):
        sv[i] += 0 if count[i] == 0 else (utility[i] / count[i])
    #sv /= n
    return sv

def cc_shap(model_name, input_data, mean_data, input_seperate, m) -> np.ndarray:
    """Compute the Shapley value by sampling complementary contributions
    """

    n = len(input_data)
    local_state = np.random.RandomState(None)
    utility = np.zeros((n + 1, n))
    count = np.zeros((n + 1, n))
    idxs = np.arange(n)
    model = load_model('./model_parameters/{}.h5'.format(model_name))
    model.compile(optimizer='adam', loss='mse')
    for _ in trange(m):
        local_state.shuffle(idxs)
        j = random.randint(1, n)
        p_1 = model.predict(generate_shap_predictive_data(input_data,mean_data,idxs[:j],input_seperate))
        p_2 = model.predict(generate_shap_predictive_data(input_data,mean_data,idxs[j:],input_seperate))
        u_1, u_2 = p_1[0][0], p_2[0][0]
        temp = np.zeros(n)
        temp[idxs[:j]] = 1
        utility[j, :] += temp * (u_1 - u_2)
        count[j, :] += temp

        temp = np.zeros(n)
        temp[idxs[j:]] = 1
        utility[n - j, :] += temp * (u_2 - u_1)
        count[n - j, :] += temp

    sv = np.zeros(n)
    for i in range(n+1):
        for j in range(n):
            sv[j] += 0 if count[i][j] == 0 else (utility[i][j] / count[i][j])
    sv /= n
    return sv

def cc_shap_XGBRegressor(model, input_data, mean_data, input_seperate, m) -> np.ndarray:
    """Compute the Shapley value by sampling complementary contributions
    """

    n = len(input_data)
    local_state = np.random.RandomState(None)
    utility = np.zeros((n + 1, n))
    count = np.zeros((n + 1, n))
    idxs = np.arange(n)

    for _ in trange(m):
        local_state.shuffle(idxs)
        j = random.randint(1, n)
        p_1 = model.predict(generate_shap_predictive_data(input_data,mean_data,idxs[:j],input_seperate))
        p_2 = model.predict(generate_shap_predictive_data(input_data,mean_data,idxs[j:],input_seperate))
        u_1, u_2 = p_1[0], p_2[0]
        temp = np.zeros(n)
        temp[idxs[:j]] = 1
        utility[j, :] += temp * (u_1 - u_2)
        count[j, :] += temp

        temp = np.zeros(n)
        temp[idxs[j:]] = 1
        utility[n - j, :] += temp * (u_2 - u_1)
        count[n - j, :] += temp

    sv = np.zeros(n)
    for i in range(n+1):
        for j in range(n):
            sv[j] += 0 if count[i][j] == 0 else (utility[i][j] / count[i][j])
    sv /= n
    return sv

def cc_shap_XGBClassifier(model, input_data, mean_data, input_seperate, m) -> np.ndarray:
    """Compute the Shapley value by sampling complementary contributions
    """

    n = len(input_data)
    local_state = np.random.RandomState(None)
    utility = np.zeros((n + 1, n))
    count = np.zeros((n + 1, n))
    idxs = np.arange(n)

    for _ in trange(m):
        local_state.shuffle(idxs)
        j = random.randint(1, n)
        p_1 = model.predict_proba(generate_shap_predictive_data(input_data,mean_data,idxs[:j],input_seperate))
        p_2 = model.predict_proba(generate_shap_predictive_data(input_data,mean_data,idxs[j:],input_seperate))
        u_1, u_2 = p_1[0][1], p_2[0][1]
        temp = np.zeros(n)
        temp[idxs[:j]] = 1
        utility[j, :] += temp * (u_1 - u_2)
        count[j, :] += temp

        temp = np.zeros(n)
        temp[idxs[j:]] = 1
        utility[n - j, :] += temp * (u_2 - u_1)
        count[n - j, :] += temp

    sv = np.zeros(n)
    for i in range(n+1):
        for j in range(n):
            sv[j] += 0 if count[i][j] == 0 else (utility[i][j] / count[i][j])
    sv /= n
    return sv

def cc_shap_DNNClassifier(model, input_data, mean_data, input_seperate, m) -> np.ndarray:
    """Compute the Shapley value by sampling complementary contributions
    """

    n = len(input_data)
    local_state = np.random.RandomState(None)
    utility = np.zeros((n + 1, n))
    count = np.zeros((n + 1, n))
    idxs = np.arange(n)

    model = load_model('./model_parameters/{}.h5'.format(model))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    for _ in trange(m):
        local_state.shuffle(idxs)
        j = random.randint(1, n)
        p_1 = model.predict(generate_shap_predictive_data(input_data,mean_data,idxs[:j],input_seperate))
        p_2 = model.predict(generate_shap_predictive_data(input_data,mean_data,idxs[j:],input_seperate))
        u_1, u_2 = p_1[0][1], p_2[0][1]
        temp = np.zeros(n)
        temp[idxs[:j]] = 1
        utility[j, :] += temp * (u_1 - u_2)
        count[j, :] += temp

        temp = np.zeros(n)
        temp[idxs[j:]] = 1
        utility[n - j, :] += temp * (u_2 - u_1)
        count[n - j, :] += temp

    sv = np.zeros(n)
    for i in range(n+1):
        for j in range(n):
            sv[j] += 0 if count[i][j] == 0 else (utility[i][j] / count[i][j])
    sv /= n
    return sv