import torch
from torch.func import jacfwd,vmap
from torchmin import Minimizer
from generateData import assembleData
import random

import torch.nn as nn

from dynamics import *

def trainNetworkFirst(net,data):
    time,position,y0 = data
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-2,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=int(0.45*500),gamma=0.1)
    epoch = 1
    while epoch<500:
        optimizer.zero_grad()  # Zero the gradients
        output = net(time,y0)  # Forward pass
        o1,o2,o3 = output[:,0],output[:,1],output[:,2]
        p1,p2,p3 = position[:,0],position[:,1],position[:,2]
        loss = torch.mean((o1-p1)**2)+1e4*torch.mean((o2-p2)**2)+torch.mean((o3-p3)**2)  # Calculate the loss
        loss.backward()
        if loss.item()<1e-6:
            epoch = 1000
        optimizer.step()
        scheduler.step()
        epoch+=1
    
def trainNetwork(y0old,y0new,net,d,lr,wd,epochs,system,data):
    
    not_converged = True
    
    ode = ode_torch(system=system)
    
    time,solution,_,derivative_fine = data #Fine integrator with previously computed initial condition

    optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.45*epochs), gamma=0.1)
    
    epoch = 0
    
    bs = min(32,len(time))
    
    while epoch<epochs and not_converged:
        random_indices = torch.randperm(len(time))[:bs]
        t_batch = time[random_indices]
        x_batch = solution[random_indices]
        v_batch = derivative_fine[random_indices]
        y0old_batch = y0old[random_indices]
        y0new_batch = y0new[random_indices]
        optimizer.zero_grad()
    
        output_pinn_old = net(t_batch,y0old_batch)
        output_pinn_new = net(t_batch,y0new_batch)
        
        delta = 1e-5 * torch.ones_like(t_batch)
        
        derivative_old = (net(t_batch+delta,y0old_batch) - net(t_batch-delta,y0old_batch)) / (2*delta)     
        derivative_new = (net(t_batch+delta,y0new_batch) - net(t_batch-delta,y0new_batch)) / (2*delta)     

        soln = output_pinn_new + x_batch - output_pinn_old
        vec = ode(t_batch,soln)
        derivative = derivative_new - derivative_old + v_batch 
        
        if system=="Robert":
            d1,d2,d3 = derivative[:,0],derivative[:,1],derivative[:,2]
            v1,v2,v3 = vec[:,0],vec[:,1],vec[:,2]
            loss = torch.mean((v1-d1)**2) + 1e4 * torch.mean((v2-d2)**2) + torch.mean((v3-d3)**2)
        else:
            loss = torch.mean((derivative - vec)**2)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch += 1
        
        if epoch>0 and epoch%50==0:
            print(f"Epoch {epoch}, Loss {loss.item()}")
        
        not_converged = loss.item()>5e-6
        
    
    net.eval();
    return not_converged