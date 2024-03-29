import numpy as np
from scripts.dynamics import vecField
from scipy.optimize import minimize,least_squares
from scipy.integrate import solve_ivp

def RK4_step(x0,dt,vecRef):
    k1 = vecRef.eval(x0)
    k2 = vecRef.eval(x0+k1*dt/2)
    k3 = vecRef.eval(x0+k2*dt/2)
    k4 = vecRef.eval(x0+k3*dt)
    return x0 + 1/6*dt*(k1+2*k2+2*k3+k4)

def solver(args,final=True):
    #This method solves the differential equation defined by f with IC u0,
    #initial time t_eval[0] and final time t_eval[1]
    
    args,vecRef = args[0],args[1]
    
    t_eval = []
    if len(args)==2:
        u0,tf=args
    if len(args)==3:
        u0,tf,t_eval=args
        
    t0 = 0.
        
    if len(t_eval)==0:
        n_steps = int(tf/vecRef.dt_fine + 1)
        time = np.linspace(t0,tf,n_steps)
    else:
        time = t_eval
    h = time[1]-time[0]
    sol = np.zeros((len(time),len(u0)))
    sol[0] = u0
    
    #sol = solve_ivp(lambda t,y : vecRef.eval(y), t_span=[0,tf], y0 = u0, t_eval=time, method='BDF', rtol=1e-8, atol=1e-8).y.T
    
    '''if vecRef.system=="Rober":
        sol = solve_ivp(lambda t,y : vecRef.eval(y), t_span = [time[0],time[-1]], y0 = u0, t_eval=time, method='BDF').y.T
    else:
        for i in range(len(time)-1):
            objective = lambda u: (u - sol[i] - h*vecRef.eval(u))
            euler_guess = sol[i]+h*vecRef.eval(sol[i])
            sol[i+1] = least_squares(objective,x0=euler_guess,method='lm').x'''
    
    if vecRef.system=="Rober" or vecRef.system=="Burger":
        for i in range(len(time)-1):
            objective = lambda u: (u - sol[i] - h*vecRef.eval(u))
            euler_guess = sol[i]+h*vecRef.eval(sol[i])
            sol[i+1] = least_squares(objective,x0=euler_guess,method='lm').x
    else:
        for i in range(len(time)-1):
            sol[i+1] = RK4_step(sol[i],h,vecRef)

    
    
    if final:
        return sol[-1]
    else:
        return sol,time