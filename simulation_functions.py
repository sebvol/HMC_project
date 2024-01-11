#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:36:09 2024

@author: sinuhe
"""

import numpy as np
from scipy import stats

#from ParticleFilter import ImportanceSampling, ParticleFilter

import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default='svg'

def simulate_sde(num_steps, dt, sigma):
    """
    Simulate a stochastic differential equation dy_t = sigma * dW_t.

    Parameters:
        - num_steps: Number of time steps.
        - dt: Time step size.
        - sigma: Volatility parameter.

    Returns:
        - A NumPy array containing the simulated process.
    """
    dW = np.random.normal(0, np.sqrt(dt), size=num_steps)
    y = np.cumsum(sigma * dW)

    return y
    
def simulate_ou_process(num_steps, dt, A, mu, V, d, initial_value=0):
    """
    Simulate a multidimensional Ornstein-Uhlenbeck process.

    Parameters:
        - num_steps: Number of time steps.
        - dt: Time step size.
        - A: Array of mean-reversion strengths for each dimension.
        - mu: Mean of the process.
        - V: Diagonal matrix of volatilities for each dimension.
        - d: Dimension of the process.
        - initial_value: Initial value of the process.

    Returns:
        - A NumPy array containing the simulated Ornstein-Uhlenbeck process with shape (num_steps, d).
    """
    ou_process = np.zeros((num_steps, d))
    ou_process[0] = initial_value

    for i in range(1, num_steps):
        dW = np.random.normal(0, np.sqrt(dt), size=(d,))
        ou_process[i] = (
            ou_process[i - 1] + A * (mu - ou_process[i - 1]) * dt + V @ dW
        )

    return ou_process

def generate_psi_path(psi, x):
    return psi * np.exp(x)

if __name__ == '__main__':
     
    num_steps = 1000
    dt = 0.01
    sigma = 0.1
    
    
    num_steps = 1000
    dt = 0.01
    A = 1.0  # Scalar mean-reversion strength
    mu = np.zeros(4)  # Mean vector of the process
    V = np.diag([0.1, 0.1, 0.1, 0.1])  # Diagonal volatility matrix
    initial_value = np.zeros(4)
    d = 4
    Psi = np.array([0.01, 0.02, 0.01, 0.024])

    x = simulate_ou_process(num_steps, dt, A, mu, V, d, initial_value)
    
    y = simulate_sde(num_steps, dt, sigma)
    psi = generate_psi_path(Psi, x)







