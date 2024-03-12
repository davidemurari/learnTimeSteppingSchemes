import numpy as np
import matplotlib.pyplot as plt
import time as time_lib


def plot_results(y0,system,time_plot,time_plot_sequential,output,network_sol,list_of_labels,total_time,time,n_x,n_t,L,vec):

    fact = 1e4*(system=="Rober") + 1. * (not system=="Rober")

    if not system=="Burger":
        
        fig = plt.figure(dpi=600)
        
        for i in range(len(y0)):
            if i==1:
                plt.plot(time_plot_sequential,output[:,i]*fact,'-',label=f"{list_of_labels[i]} reference")
                plt.plot(time_plot,network_sol[i]*fact,'--',label=f"{list_of_labels[i]} parareal")
            else:
                plt.plot(time_plot_sequential,output[:,i],'-',label=f"{list_of_labels[i]} reference")
                plt.plot(time_plot,network_sol[i],'--',label=f"{list_of_labels[i]} parareal")
                
        plt.legend()
        plt.title(f"Comparison of solutions, n_x={n_x}, n_t={n_t}, L={L},\n computational time = {np.round(total_time,2)}, len(time)={len(time)}")
        timestamp = time_lib.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"savedPlots/pararealPlot_{system}_{timestamp}.pdf")
        plt.show();
        
    if system=="Brusselator":
        
        fig = plt.figure(dpi=600)
        
        plt.plot(network_sol[0],network_sol[1],'k-',label="parareal")
        plt.plot(output[:,0],output[:,1],'r--',label="reference")
        
        plt.legend()
        plt.title(f"Orbits in the phase space, n_x={n_x}, n_t={n_t}, L={L},\n computational time = {np.round(total_time,2)}, len(time)={len(time)}")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        timestamp = time_lib.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"savedPlots/orbits_{system}_{timestamp}.pdf")
        plt.show();
        
    if system=="Arenstorf":
        
        fig = plt.figure(dpi=600)
        
        plt.plot(network_sol[0],network_sol[2],'k-',label="parareal")
        plt.plot(output[:,0],output[:,2],'r--',label="reference")
        
        plt.legend()
        plt.title(f"Orbits in the phase space, n_x={n_x}, n_t={n_t}, L={L},\n computational time = {np.round(total_time,2)}, len(time)={len(time)}")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        timestamp = time_lib.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"savedPlots/orbits_{system}_{timestamp}.pdf")
        plt.show();

    if system=="Burger":
                
        fig = plt.figure(dpi=600,figsize=(10,7))
        # Plotting the solution
        for i in range(0, len(time_plot), 5):
            plt.plot(vec.x, output[i],'k-',label=f't={time_plot[i]:.2f}')
        for i in range(0, len(time_plot), 5):
            plt.plot(vec.x, network_sol[:,i],'r--',label=f't={time_plot[i]:.2f}')

        plt.title(f'Solution of the Viscous Burger\'s Equation, n_x={n_x}, n_t={n_t}, L={L},\n computational time = {np.round(total_time,2)}, len(time)={len(time)}')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x,t)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        timestamp = time_lib.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"savedPlots/solution_curves_{system}_{timestamp}.pdf")

        plt.show()

        from mpl_toolkits.mplot3d import Axes3D

        # Creating the meshgrid for x and t
        T, X = np.meshgrid(time_plot, vec.x)

        # Creating the figure

        fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))  # 1 row, 2 columns of 3D subplots

        # Plotting the surface
        surf = axs[0].plot_surface(X, T, output.T, cmap='viridis', edgecolor='none')
        surf = axs[1].plot_surface(X, T, network_sol, cmap='viridis', edgecolor='none')

        fig.suptitle(f'Comparison of the surface plots, n_x={n_x}, n_t={n_t}, L={L},\n computational time = {np.round(total_time,2)}, len(time)={len(time)}')

        # Labels and title
        axs[0].set_xlabel(r'$x$')
        axs[0].set_ylabel(r'$t$')
        axs[0].set_zlabel(r'$u(x, t)$')
        axs[0].set_title("Reference solution")

        axs[1].set_xlabel(r'$x$')
        axs[1].set_ylabel(r'$t$')
        axs[1].set_zlabel(r'$u(x, t)$')
        axs[1].set_title("Network prediction")

        timestamp = time_lib.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"savedPlots/solution_surface_{system}_{timestamp}.pdf")

        # Show plot
        plt.show()