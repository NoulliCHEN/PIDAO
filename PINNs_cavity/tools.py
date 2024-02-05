# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:00:36 2022

@author: Shengze Cai
"""

import math
import numpy as np
import matplotlib.pyplot as plt



def LHSample(D, bounds, N):
    # """
    # :param D: Number of parameters
    # :param bounds:  [[min_1, max_1],[min_2, max_2],[min_3, max_3]](list)
    # :param N: Number of samples
    # :return: Samples
    # """
    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N
    for i in range(D):
        for j in range(N):
            temp[j] = np.random.uniform(low=j * d, high=(j + 1) * d, size=1)[0]
        np.random.shuffle(temp)
        for j in range(N):
            result[j, i] = temp[j]
    # Stretching the sampling
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]
    if np.any(lower_bounds > upper_bounds):
        print('Wrong value bound')
        return None
    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result, (upper_bounds - lower_bounds), out=result),
           lower_bounds,
           out=result)
    return result



   
    
    
    
    
    
    
    
