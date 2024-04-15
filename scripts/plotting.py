import numpy as np
import matplotlib.pyplot as plt
import time as time_lib
import matplotlib

#Setting the plotting parameters
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams['font.size']= 45
matplotlib.rcParams['font.family']= 'ptm' #'Times New Roman

def plot_results(y0,coarse_approx,networks,system,time_plot,time_plot_sequential,output,network_sol,list_of_labels,total_time,time,n_x,n_t,L,vec,number_iterates=1,btype=None):

    fact = 1e4*(system=="Rober") + 1. * (not system=="Rober")

    if not system=="Burger":
        
        fig = plt.figure(figsize=(20,10))
        
        for i in range(len(y0)):
            if i==1:
                plt.plot(time_plot_sequential,output[:,i]*fact,'-',label=f"{list_of_labels[i]} reference",linewidth=5)
                plt.plot(time_plot,network_sol[i]*fact,'--',label=f"{list_of_labels[i]} parareal",linewidth=5)
            else:
                plt.plot(time_plot_sequential,output[:,i],'-',label=f"{list_of_labels[i]} reference",linewidth=5)
                plt.plot(time_plot,network_sol[i],'--',label=f"{list_of_labels[i]} parareal",linewidth=5)

        '''time_coarse = np.linspace(0,time_plot[-1],len(networks)+1)
        for j in range(len(networks)):
            for i in range(len(y0)):
                plt.plot(time_coarse,coarse_approx[:,i],'co')'''
                
        plt.legend(fontsize=25,loc='center left', bbox_to_anchor=(1, 0.5))

        title = f"Comparison of solutions, $C$={n_x}, $H$={L},\n Median computational time over {number_iterates} iterates: {np.round(total_time,2)}s" if number_iterates>1 \
                else f"Comparison of solutions, $C$={n_x}, $H$={L},\n Computational time: {np.round(total_time,2)}s"
        plt.title(title) 
        plt.xlabel(r'$t$')
        #timestamp = time_lib.strftime("%Y%m%d_%H%M%S")
        #plt.savefig(f"savedPlots/pararealPlot_{system}_{timestamp}.pdf")
        name_plot = f"savedPlots/ELM_pararealPlot_{system}.pdf" if system=="SIR" else f"savedPlots/pararealPlot_{system}.pdf"
        plt.savefig(name_plot,bbox_inches='tight')
        plt.show();
        
    if system=="Brusselator":
        
        fig = plt.figure(figsize=(10,10))
        
        plt.plot(output[:,0],output[:,1],'k-',label="reference",linewidth=5)
        plt.plot(network_sol[0],network_sol[1],'r--',label="parareal",linewidth=5)

        
        plt.legend(fontsize=25)
        plt.title(f"Orbits in the phase space")
        plt.xlabel(list_of_labels[0])
        plt.ylabel(list_of_labels[1])
        #timestamp = time_lib.strftime("%Y%m%d_%H%M%S")
        #plt.savefig(f"savedPlots/orbits_{system}_{timestamp}.pdf")
        plt.savefig(f"savedPlots/orbits_{system}.pdf",bbox_inches='tight')
        plt.show();
        
    if system=="Arenstorf":
        
        fig = plt.figure(figsize=(10,10))
        
        plt.plot(output[:,0],output[:,2],'k-',label="reference",linewidth=5)
        plt.plot(network_sol[0],network_sol[2],'r--',label="parareal",linewidth=5)
        plt.legend(fontsize=25)
        #plt.legend()
        plt.title(f"Orbits in the phase space")
        plt.xlabel(list_of_labels[0])
        plt.ylabel(list_of_labels[1])
        #timestamp = time_lib.strftime("%Y%m%d_%H%M%S")
        #plt.savefig(f"savedPlots/orbits_{system}_{timestamp}.pdf")
        plt.savefig(f"savedPlots/orbits_{system}.pdf",bbox_inches='tight')
        plt.show();

    if system=="Burger":
                
        fig = plt.figure(figsize=(20,10))
        # Plotting the solution
        
        step = len(time_plot)//10
        
        for i in range(0,len(time_plot),step):
                if i==0:
                    plt.plot(vec.x, output[i],'k-',linewidth=5,label="reference")
                else:
                    plt.plot(vec.x, output[i],'k-',linewidth=5)
        for i in range(0,len(time_plot),step):
                if i==0:
                    plt.plot(vec.x, network_sol[:,i],'r--',linewidth=5,label="parareal")
                else:
                    plt.plot(vec.x, network_sol[:,i],'r--',linewidth=5)
        #for i in range(len(networks)):
        #    plt.plot(vec.x,coarse_approx[i],'co')

        title = f"Comparison of solutions, $C$={n_x}, $H$={L},\n Median computational time over {number_iterates} iterates: {np.round(total_time,2)}s" if number_iterates>1 \
                else f"Comparison of solutions, $C$={n_x}, $H$={L},\n Computational time: {np.round(total_time,2)}s"
        
        plt.legend(fontsize=25,loc='center left', bbox_to_anchor=(1, 0.5))

        plt.title(title)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x,t)$')
        #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        #timestamp = time_lib.stftime("%Y%m%d_%H%M%S")
        #plt.savefig(f"savedPlots/solution_curves_{system}_{timestamp}.pdf")
        plt.savefig(f"savedPlots/solution_curves_{system}_{btype}.pdf",bbox_inches='tight')

        plt.show()

        from mpl_toolkits.mplot3d import Axes3D

        # Creating the meshgrid for x and t
        T, X = np.meshgrid(time_plot, vec.x)

        # Creating the figure

        fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(20,10))  # 1 row, 2 columns of 3D subplots

        # Plotting the suface
        surf = axs[0].plot_surface(X, T, output.T, cmap='viridis', edgecolor='none')
        surf = axs[1].plot_surface(X, T, network_sol, cmap='viridis', edgecolor='none')

        fig.suptitle(f'Comparison of the surface plots')

        # Labels and title
        #axs[0].set_xlabel(r'$x$')
        #axs[0].set_ylabel(r'$t$')
        #axs[0].set_zlabel(r'$u(x, t)$')
        axs[0].set_title("Reference solution")

        #axs[1].set_xlabel(r'$x$')
        #axs[1].set_ylabel(r'$t$')
        #axs[1].set_zlabel(r'$u(x, t)$')
        axs[1].set_title("Network prediction")

        #timestamp = time_lib.strftime("%Y%m%d_%H%M%S")
        #plt.savefig(f"savedPlots/solution_surface_{system}_{timestamp}.pdf")
        plt.savefig(f"savedPlots/solution_surface_{system}_{btype}.pdf",bbox_inches='tight')

        # Show plot
        plt.show()