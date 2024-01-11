from scipy.integrate import solve_ivp
from dynamics import *

def fineIntegratorParallel(coarse_values_parareal,i,system,dt,ode):
        return solve_ivp(ode,[0,dt], y0=coarse_values_parareal[i], method = 'RK45', atol=1e-6, rtol=1e-6, args=(system,)).y.T[-1]