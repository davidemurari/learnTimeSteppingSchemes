import torch
import numpy as np

from dynamics import *
from scipy.integrate import solve_ivp

from torchdiffeq import odeint_adjoint as odeint


def assembleData(y0,t_span,system,derivative=True):
        
    time = t_span
    
    output = solve_ivp(ode, [t_span[0],t_span[-1]], y0, t_eval=time,method = 'BDF', atol=1e-10, rtol=1e-10, args=(system,)).y
    
    if derivative:
        delta = 1e-5 * np.ones_like(time)
        output_b = solve_ivp(ode, [t_span[0]+1e-5,t_span[-1]+1e-5], y0, t_eval=time+delta,method = 'BDF', atol=1e-10, rtol=1e-10, args=(system,)).y.T
        output_f = solve_ivp(ode, [t_span[0]-1e-5,t_span[-1]-1e-5], y0, t_eval=time-delta,method = 'BDF', atol=1e-10, rtol=1e-10, args=(system,)).y.T
    sol = torch.from_numpy(output.T.astype(np.float32)).reshape(-1,3) #.y.T #num_times x d
    time = torch.from_numpy(time.astype(np.float32)).unsqueeze(1)
    
    y0 = torch.from_numpy(y0.astype(np.float32)).unsqueeze(0).repeat(len(time),1)
    
    if derivative:
        derivative_fine = torch.from_numpy(((output_f-output_b) / (2*delta.reshape(-1,1))).astype(np.float32)).reshape(-1,3)
        return time,sol,y0,derivative_fine
    else:
        return time,sol,y0

    