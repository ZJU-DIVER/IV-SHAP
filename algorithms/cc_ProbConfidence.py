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

from sqhsrc.utils.tools import (split_permutation_num, split_permutation, split_num,
                    power_set)

def ccshap_prob_confidence(model_name, input_data, mean_data, input_seperate, initial_m, local_m) -> np.ndarray:
    """Compute the Shapley value by sampling complementary contributions based on Neyman
    """
    n = len(input_data)
    sv = np.zeros(n)
    utility = [[[] for _ in range(n)] for _ in range(n)]
    var = np.zeros((n, n))
    local_state = np.random.RandomState(None)
    coef = [comb(n - 1, s) for s in range(n)]
    model = load_model('./model_parameters/{}.h5'.format(model_name))
    model.compile(optimizer='adam', loss='mse')
    # initialize
    cnt = 0
    while True:
        temp_count = cnt
        for i in trange(n):
            idxs = [_ for _ in range(i)] + [_ for _ in range(i + 1, n)]
            for j in range(n):
                if len(utility[i][j]) >= initial_m:
                    continue
                local_state.shuffle(idxs)
                cnt += 1
                p_1 = model.predict(generate_shap_predictive_data(input_data,mean_data,idxs[:j]+[i],input_seperate))
                p_2 = model.predict(generate_shap_predictive_data(input_data,mean_data,idxs[j:],input_seperate))
                u_1, u_2 = p_1[0][0], p_2[0][0]
                utility[i][j].append(u_1 - u_2)
                for l in range(n - 1):
                    if l < j:
                        utility[idxs[l]][j].append(u_1 - u_2)
                    else:
                        utility[idxs[l]][n - j - 2].append(u_2 - u_1)
        if cnt == temp_count:
            break
        
    # compute allocation
    for i in range(n):
        for j in range(n):
            var[i][j] = np.var(utility[i][j])
            var[i][j] = 0 if var[i][j] == 0 else var[i][j] * len(
                utility[i][j]) / (len(utility[i][j]) - 1)

    new_utility = np.zeros((n, n))
    count = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            new_utility[i][j] = np.sum(utility[i][j])
            count[i][j] = len(utility[i][j])

    local_m -= cnt
    
    print(local_m)
    cals = np.zeros(n)
    for _ in range(local_m):
        sigma_j = np.zeros(n)
        sigma_n_j = np.zeros(n)
        for k in range(math.ceil(n / 2) - 1, n):
            for i in range(n):
                sigma_j[k] += var[i][k] / count[i][k]
                if n - k - 2 < 0:
                    sigma_n_j[k] += 0
                else:
                    sigma_n_j[k] += var[i][n - k - 2] / count[i][n-k-2]
        
        sta = math.ceil(n / 2) - 1
        interval_distance_sum = 0 
        for k in range(math.ceil(n / 2) - 1, n):
            interval_distance_sum += sigma_j[k] + sigma_n_j[k]
        prob = np.zeros(n)   
        for k in range(math.ceil(n / 2) - 1, n):
            prob[k] = (sigma_j[k] + sigma_n_j[k])/interval_distance_sum

        j = np.random.choice(range(sta, n), p=prob[sta:n])
        cals[j] += 1
        idxs = np.arange(n)
        local_state.shuffle(idxs)
        p_1 = model.predict(generate_shap_predictive_data(input_data,mean_data,idxs[:j + 1],input_seperate))
        p_2 = model.predict(generate_shap_predictive_data(input_data,mean_data,idxs[j + 1:],input_seperate))

        u_1, u_2 = p_1[0][0], p_2[0][0]

        for k in range(n):
            if k <= j:
                average = (new_utility[idxs[k]][j]+u_1-u_2)/(count[idxs[k]][j]+1)
                var[idxs[k]][j] = (count[idxs[k]][j]*var[idxs[k]][j]+count[idxs[k]][j]*(u_1-u_2-average)**2/(count[idxs[k]][j]+1)) / (count[idxs[k]][j]+1)
            else:
                average = (new_utility[idxs[k]][n-j-2]+u_2-u_1)/(count[idxs[k]][n-j-2]+1)
                var[idxs[k]][n-j-2] = (count[idxs[k]][n-j-2]*var[idxs[k]][n-j-2]+count[idxs[k]][n-j-2]*(u_2-u_1-average)**2/(count[idxs[k]][n-j-2]+1)) / (count[idxs[k]][n-j-2]+1)
        
        temp = np.zeros(n)
        temp[idxs[:j + 1]] = 1
        new_utility[:, j] += temp * (u_1 - u_2)
        count[:, j] += temp

        temp = np.zeros(n)
        temp[idxs[j + 1:]] = 1
        new_utility[:, n - j - 2] += temp * (u_2 - u_1)
        count[:, n - j - 2] += temp

    for i in range(n):
        for k in range(n):
            sv[i] += 0 if count[i][k] == 0 else new_utility[i][k] / count[i][k]
        sv[i] /= n
    return sv

