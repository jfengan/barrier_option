# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:18:05 2021

@author: Administrator
"""

import numpy as np
import time
import torch
from scipy.stats import norm


class Simulator:
    @staticmethod
    def simulate_pseudo(spot, r, q, sigma, dt, num_paths, time_steps):
        np.random.seed(1234)
        half_path = int(num_paths / 2) + 1
        sqrt_var = sigma * np.sqrt(dt)
        st = spot * np.ones((num_paths, time_steps+1))
        # start = timeit.default_timer()
        simu = np.random.normal(0, 1, (half_path, time_steps))
        anti_simu = -simu
        simulation = np.concatenate((simu, anti_simu))[:num_paths,:]
        growth = (r - q - 0.5*sigma*sigma) * dt + sqrt_var * simulation
        factor = np.exp(growth)
        for i in range(1, time_steps+1):
            st[:, i] = st[:, i-1] * factor[:, i-1]
        return st
    
    @staticmethod
    def simulate_sobol(spot, r, q, sigma, dt, num_paths, time_steps):
        sqrt_var = sigma * np.sqrt(dt)
        st = spot * np.ones((num_paths, time_steps+1))
        sobol_engine = torch.quasirandom.SobolEngine(dimension=time_steps, scramble=True, seed=int(time.time()))
        sobol_rng = np.array(sobol_engine.draw(num_paths, dtype=torch.float64))
        simulation = norm.ppf(sobol_rng)
        growth = (r - q - 0.5*sigma*sigma) * dt + sqrt_var * simulation
        factor = np.exp(growth)
        for i in range(1, time_steps+1):
            st[:, i] = st[:, i-1] * factor[:, i-1]
        return st
