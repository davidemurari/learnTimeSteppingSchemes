import torch
from torch.func import jacfwd,vmap

from scripts.dynamics import *

def trainNetwork(dt,net,d,lr,wd,epochs,system,dtype=torch.float32):
    
    ode = ode_torch(system=system)
    
    optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.45*epochs), gamma=0.1)
    #optimizer = torch.optim.LBFGS(net.parameters())
    
    epoch = 0
    
    bs = 5000
    is_not_converged = True
    while epoch<epochs and is_not_converged:
        
        t_batch = torch.rand((bs,1),dtype=dtype) * dt
        
        if system=="Robert":
            x = torch.rand(bs,1,dtype=dtype)
            y = torch.rand(bs,1,dtype=dtype) * 1e-4
            z = torch.rand(bs,1,dtype=dtype)
            y0new_batch = torch.cat((x,y,z),dim=1)
            #y0new_batch = torch.tensor([[1.,0.,0.]],dtype=dtype).repeat(bs,1)
        elif system=="SIR":
            y0new_batch = torch.rand(bs,d,dtype=dtype)
        
        def closure():   
            optimizer.zero_grad()
        
            def func(t,x):
                t = t.reshape(-1,1)
                x = x.reshape(-1,d)
                
                return net(t,x)

            dfunc = lambda t,x : vmap(jacfwd(func,argnums=0))(t,x)[:,0,:,0]
        
            soln = func(t_batch,y0new_batch)
            derivative = dfunc(t_batch,y0new_batch)
            vec = ode(t_batch,soln)
            
            if system=="Robert":
                d1,d2,d3 = derivative[:,0],derivative[:,1],derivative[:,2]
                v1,v2,v3 = vec[:,0],vec[:,1],vec[:,2]
                loss = torch.mean((v1-d1)**2) + 1e6*torch.mean((v2-d2)**2) 
                loss += torch.mean((v3-d3)**2)
            else:
                loss = torch.mean((derivative-vec)**2)
            #if torch.is_grad_enabled():
            loss.backward()
            return loss
        
        optimizer.step(closure)
        scheduler.step()
        epoch += 1
        loss = closure()
        is_not_converged = loss.item()>1e-8
        
        if epoch>0 and epoch%50==0:
            print(f"Epoch {epoch}, Loss {loss.item()}")
    net.eval();
    return loss.item()