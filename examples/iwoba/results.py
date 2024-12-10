# type: ignore
import torch
import matplotlib.pyplot as plt
import os
from borc2.utilities import tic, toc 

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
    colours = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']
    fontsize = 16 

    error = torch.abs(data - y_optimal)
    mean_error = error.mean(dim=0)
    std_error = error.std(dim=0)
    upper_bound_error = mean_error + 1.96 * std_error
    lower_bound_error = mean_error - 1.96 * std_error
    mean_error = mean_error.numpy()
    upper_bound_error = upper_bound_error.numpy()
    lower_bound_error = lower_bound_error.numpy()

    plt.figure(figsize=(6, 6))
    label = fr'$\alpha_{{\text{{{names[d]}}}}}$' if d != 1 else fr'$\tilde\alpha_{{\text{{{names[d]}}}}}$'
    plt.plot(x, mean_error, '-', color=colours[d], linewidth=2, label=label)
    plt.scatter(x, mean_error, color=colours[d], s=30, edgecolor='black', label="Data points")
    plt.fill_between(x, lower_bound_error, upper_bound_error, color=colours[d], alpha=0.2, label="95% CI")
    plt.axhline(y=0.0, color='gray', linestyle='--', linewidth=1, label="Baseline")

    plt.xlabel("Iterations", fontsize=fontsize)
    plt.ylabel(r"Absolute Error $\quad$ $y^*-\text{E}[\text{f}(x^*, \xi)]$", fontsize=fontsize)
    # plt.ylim([-500, 1500])
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=fontsize-2, loc='upper left', frameon=True)
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
    N = 10 # test every nth iters
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

    # ======================================== #  
    # prestress   
    # ======================================== #  
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

