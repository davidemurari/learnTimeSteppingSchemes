import torch
from generateData import assembleData

from torch.func import jacfwd,vmap
from torchmin import Minimizer

import random

import torch.nn as nn

from dynamics import *

def trainNetwork(y0,t_0,t_1,net,d,lr,wd,epochs,system = "SIR",is_pinn=True,is_reg=True):
    
    time,position = assembleData(t_0,t_1,y0,system)
    
    time = time - t_0
    
    optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.45*epochs), gamma=0.1)
    
    if is_pinn==False and is_reg==False:
        is_reg = True
        print("Regression loss set to true so there is something to optimise for")
    
    for epoch in range(epochs):
        
        
    
        optimizer.zero_grad()
    
        loss = 0
        
        if is_pinn:
            
            time_pinn = torch.rand(100)*(t_1-t_0)
            time_pinn = time_pinn.unsqueeze(1)
            IC_pinn = torch.from_numpy(y0.astype(np.float32)).reshape(1,d).repeat(len(time_pinn),1)
            
            def func(t,x):
                t = t.reshape(-1,1)
                x = x.reshape(-1,d)
                return net(t,x)

            dfunc = lambda t,x : vmap(jacfwd(func,argnums=0))(t,x)[:,0,:,0]
            
            output_pinn = func(time_pinn,IC_pinn)
            derivative = dfunc(time_pinn,IC_pinn)
            
            if system=="Robert":
                d1,d2,d3 = derivative[:,0],derivative[:,1],derivative[:,2]
                vec = ode_torch(time_pinn,output_pinn,system)
                v1,v2,v3 = vec[:,0],vec[:,1],vec[:,2]
                loss += torch.mean((v1-d1)**2) + 1e6 * torch.mean((v2-d2)**2) + torch.mean((v3-d3)**2)
            else:
                loss += torch.mean((derivative - ode_torch(time_pinn,output_pinn,system))**2)
        
        if is_reg:
            
            M = min(len(time),16)
            idx = random.sample(range(0,len(time)), M)
            
            time_reg = time[idx]
            
            IC = torch.from_numpy(y0.astype(np.float32)).reshape(1,d).repeat(len(time_reg),1)
            output = net(time_reg,IC)
            if system=="Robert":
                p1,p2,p3 = output[:,0], output[:,1], output[:,2]
                t1,t2,t3 = position[idx,0], position[idx,1], position[idx,2]
                loss += torch.mean((p1-t1)**2) + 1e6 * torch.mean((p2-t2)**2) + torch.mean((p3-t3)**2)
            else:
                loss += torch.mean(((output - position))**2)
            
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch>0 and epoch%100==0:
            print(f"Epoch {epoch}, Loss {loss.item()}")
        if loss.item()<1e-9:
            epoch = epochs + 5
            break
    
    net.eval();