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

def cc_shap(model_name, input_data, mean_data, input_seperate, m, proc_num=1) -> np.ndarray:
    """Compute the Shapley value by sampling complementary contributions
    """
    if proc_num < 0:
        raise ValueError('Invalid proc num.')

    n = len(input_data)
    args = split_permutation_num(m, proc_num)
    pool = Pool()
    func = partial(_cc_shap_task, model_name,input_data,mean_data,input_seperate)
    ret = pool.map(func, args)
    pool.close()
    pool.join()

    sv = np.zeros(n)
    utility = np.zeros((n + 1, n))
    count = np.zeros((n + 1, n))
    for r in ret:
        utility += r[0]
        count += r[1]
    for i in range(n + 1):
        for j in range(n):
            sv[j] += 0 if count[i][j] == 0 else (utility[i][j] / count[i][j])
    sv /= n
    return sv

def _cc_shap_task(model_name, input_data, mean_data, input_seperate, local_m) -> np.ndarray:
    """Compute the Shapley value by sampling local_m complementary contributions
    """
    n = len(input_data)
    local_state = np.random.RandomState(None)
    utility = np.zeros((n + 1, n))
    count = np.zeros((n + 1, n))
    idxs = np.arange(n)

    for _ in trange(local_m):
        local_state.shuffle(idxs)
        j = random.randint(1, n)
        model = load_model('./model_parameters/{}.h5'.format(model_name))
        model.compile(optimizer='adam', loss='mse')
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

    return utility, count