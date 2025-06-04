# type: ignore
import torch
import matplotlib.pyplot as plt
import os
from borc2.utilities import tic, toc 
import numpy as np

import warnings
warnings.filterwarnings("ignore", message=r".*You are using*")  
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

from branin_rs import bayesopt as bayesopt_rs
from branin_ei import bayesopt as bayesopt_ei 
from branin_er import bayesopt as bayesopt_er 
# from prestress_rs import bayesopt as bayesopt_rs
# from prestress_ei import bayesopt as bayesopt_ei
# from prestress_er import bayesopt as bayesopt_er

# Author: James Whiteley (github.com/jamesalexwhiteley)

def plotdata(d, x, data):
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'{name}_bayesopt.png')
    
    data = data[:, ::N]
    # colours = ['#d2691e', '#55b7af', '#4191d5']
    colours = ['#0072B2', '#009E73', '#D55E00']
    fontsize = 16 

    # Data for plotting 
    error = torch.abs(data - y_optimal)

    # fig definition
    plt.figure(figsize=(6.25, 6))
    label = fr'$\alpha_{{\text{{{names[d]}}}}}$' if d != 1 else fr'$\tilde\alpha_{{\text{{{names[d]}}}}}$'

    # Individual runs 
    n_show = min(15, data.shape[0])
    show_alpha = 0.15 if data.shape[0] > 10 else 0.3
    for i in range(n_show):
        individual_error = error[i].numpy()
        label_run = 'Sample runs' if i == 0 else None  # Only label the first one
        plt.plot(x, individual_error, '-', color=colours[d], 
                alpha=show_alpha, linewidth=1, label=label_run)

    # Percentile bands
    p25 = torch.quantile(error, 0.25, dim=0).numpy()
    p50 = torch.quantile(error, 0.50, dim=0).numpy()
    p75 = torch.quantile(error, 0.75, dim=0).numpy()
    plt.fill_between(x, p25, p75, color=colours[d], alpha=0.2, label=f'{label} (IQR)')
    plt.plot(x, p50, color=colours[d], linewidth=2.5, alpha=1.0, label=f'{label} (median)')
    
    # Markers
    plt.scatter(x, p50, color=colours[d], s=30, edgecolor=colours[d], label="Data points")

    # plt.xlim([5, 50])
    # plt.ylim([-1000, 2500])

    # Background lines 
    plt.axhline(y=0.0, color='#666666', linestyle='-', linewidth=1, alpha=0.7)
    plt.grid(True, linestyle='-', alpha=0.2, color='gray')

    # Labels 
    plt.xlabel("Iterations", fontsize=fontsize)
    plt.ylabel(r"Absolute Error $\quad$ $y^*-\text{E}[\text{f}(x^*, \xi)]$", fontsize=fontsize)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=fontsize-2, loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.show()

def plotfig(DATA): 
    print(f"No. of runs of each algorithm: {DATA.size(1)}")
    x = torch.linspace(1, DATA.size(2), int(DATA.size(2)/N))
    for d, data in enumerate(DATA):
        plotdata(d, x, data)
                
def scriptloader(batch_n, fun, DATA):
    for i in range(RUNS):
        print(f"Batch No. {batch_n}, Run No. {i}")
        try:
            _, data = fun(NINITIAL, ITERS, N) 
        except Exception as e:
            print(f"Error occurred in run {i} with function {fun}: {e}")
            data = DATA[batch_n, i-1, :]
            pass 
        DATA[batch_n, i, :] = data 

if __name__ == "__main__":

    # ======================================== #  
    # branin   
    # ======================================== #  
    functions = [bayesopt_rs, bayesopt_er, bayesopt_ei] 
    names = ['RS', 'EIxPF-WSE', 'EIxPF']
    name = 'branin'
    y_optimal = 8330
    BATCHS = len(functions) 
    RUNS = 10 # runs (of each algorithm)
    ITERS = 50 # iterations (on each run)
    NINITIAL = 10 # initial points (on each run)
    N = 2 # test every nth iters
    DATA = torch.ones(BATCHS, RUNS, ITERS)

    # filepath 
    if not os.path.exists('data'): 
        os.makedirs('data') 
    file_path = os.path.join('data', 'branin.pt') 

    # # scripts 
    # for j, fun in enumerate(functions): 
    #     tic() 
    #     scriptloader(j, fun, DATA)  
    #     torch.save(DATA, file_path) 
    #     toc() 

    # load data  
    DATA = torch.load(file_path) 
    plotfig(DATA) 

    # # ======================================== #  
    # # prestress   
    # # ======================================== #  
    # # functions = [bayesopt_rs, bayesopt_rei, bayesopt_ei]
    # names = ['RS', 'EIxPF-WSE', 'EIxPF']
    # name = 'prestressed'
    # y_optimal = -830
    # # BATCHS = len(functions)
    # RUNS = 50 # runs (of each algorithm)
    # ITERS = 50 # iterations (on each run)
    # NINITIAL = 5 # initial points (on each run)
    # N = 2 # test every nth iter
    # # DATA = torch.ones(BATCHS, RUNS, ITERS)

    # # filepath 
    # if not os.path.exists('data'): 
    #     os.makedirs('data') 
    # file_path = os.path.join('data', 'prestress.pt') 

    # # # scripts 
    # # for j, fun in enumerate(functions): 
    # #     tic() 
    # #     scriptloader(j, fun, DATA) 
    # #     torch.save(DATA, file_path) 
    # #     toc() 
 
    # # load data 
    # DATA = torch.load(file_path) 
    # plotfig(DATA) 

