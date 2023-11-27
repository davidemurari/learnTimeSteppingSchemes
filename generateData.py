import torch
import numpy as np

from dynamics import *
from scipy.integrate import solve_ivp


def assembleData(t_0,t_1,y0,system):
    
    num_times = 50

    time = np.linspace(t_0,t_1,num_times)
    
    output = solve_ivp(ode,[t_0,t_1], y0, t_eval=time, method = 'BDF', atol=1e-10, rtol=1e-10, args=(system,))
    sol = output.y.T #num_times x d
    time = output.t
    
    t_torch, sol_torch = torch.from_numpy(time.astype(np.float32)).unsqueeze(1), torch.from_numpy(sol.astype(np.float32))
    
    return t_torch,sol_torch


    