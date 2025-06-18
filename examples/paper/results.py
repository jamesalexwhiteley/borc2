# # type: ignore
# import torch
# import matplotlib.pyplot as plt
# import os
# from borc2.utilities import tic, toc 
# import numpy as np

# import warnings
# warnings.filterwarnings("ignore", message=r".*You are using*")  
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Arial']

# # from branin_rs import bayesopt as bayesopt_rs
# # from branin_ei import bayesopt as bayesopt_ei 
# # from branin_er import bayesopt as bayesopt_er 
# from prestress_rs import bayesopt as bayesopt_rs
# from prestress_ei import bayesopt as bayesopt_ei
# from prestress_er import bayesopt as bayesopt_er

# # Author: James Whiteley (github.com/jamesalexwhiteley)

# def plotdata(d, x, data):
#     output_dir = 'figures'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     output_path = os.path.join(output_dir, f'{name}_bayesopt.png')
    
#     data = data[:, ::N]
#     # colours = ['#d2691e', '#55b7af', '#4191d5']
#     colours = ['#0072B2', '#009E73', '#D55E00']
#     fontsize = 16 

#     # Data for plotting 
#     error = torch.abs(data - y_optimal)

#     # fig definition
#     plt.figure(figsize=(6.25, 6))
#     label = fr'$\alpha_{{\text{{{names[d]}}}}}$' if d != 1 else fr'$\tilde\alpha_{{\text{{{names[d]}}}}}$'

#     # Individual runs 
#     n_show = min(15, data.shape[0])
#     show_alpha = 0.15 if data.shape[0] > 10 else 0.3
#     for i in range(n_show):
#         individual_error = error[i].numpy()
#         label_run = 'Sample runs' if i == 0 else None  # only label first 
#         plt.plot(x, individual_error, '-', color=colours[d], 
#                 alpha=show_alpha, linewidth=1, label=label_run)

#     # Percentile bands
#     p25 = torch.quantile(error, 0.25, dim=0).numpy()
#     p50 = torch.quantile(error, 0.50, dim=0).numpy()
#     p75 = torch.quantile(error, 0.75, dim=0).numpy()
#     plt.fill_between(x, p25, p75, color=colours[d], alpha=0.2, label=f'{label} (IQR)')
#     plt.plot(x, p50, color=colours[d], linewidth=2.5, alpha=1.0, label=f'{label} (median)')
    
#     # Markers
#     plt.scatter(x, p50, color=colours[d], s=30, edgecolor=colours[d], label="Data points")
#     # plt.ylim([0, 2000])

#     # Background lines 
#     plt.axhline(y=0.0, color='#666666', linestyle='-', linewidth=1, alpha=0.7)
#     plt.grid(True, linestyle='-', alpha=0.2, color='gray')

#     # Labels 
#     plt.xlabel("Iterations", fontsize=fontsize)
#     plt.ylabel(r"Absolute Error $\quad$ $y^*-\text{E}[\text{f}(x^*, \xi)]$", fontsize=fontsize)
#     plt.grid(visible=True, linestyle='--', alpha=0.6)
#     plt.legend(fontsize=fontsize-2, loc='upper right', frameon=True, fancybox=True, shadow=True)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=600)
#     plt.show()

# def plotfig(DATA): 
#     print(f"No. of runs of each algorithm: {DATA.size(1)}")
#     x = torch.linspace(1, DATA.size(2), int(DATA.size(2)/N))
#     for d, data in enumerate(DATA):
#         plotdata(d, x, data)

# def scriptloader(batch_n, fun, DATA):
#     for i in range(RUNS):
#         print(f"Batch No. {batch_n}, Run No. {i}")
#         try:
#             # Add memory cleanup
#             torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
#             _, data = fun(NINITIAL, ITERS, N) 
#             DATA[batch_n, i, :] = data
            
#             # Save after each run to avoid losing progress
#             if i % 5 == 0:  # Save every 5 runs
#                 torch.save(DATA, file_path)
#                 # print(f"Checkpoint saved at run {i}")
                
#         except Exception as e:
#             print(f"Error occurred in run {i} with function {fun.__name__}: {e}")
#             import traceback
#             traceback.print_exc()  # Print full error traceback
            
#             # Use previous run's data as fallback
#             if i > 0:
#                 DATA[batch_n, i, :] = DATA[batch_n, i-1, :]
#             else:
#                 DATA[batch_n, i, :] = torch.zeros(ITERS)  # or some default

# if __name__ == "__main__":

#     # # ======================================== #  
#     # # branin   
#     # # ======================================== #  
#     # functions = [bayesopt_rs, bayesopt_er, bayesopt_ei] 
#     # names = ['RS', 'EIxPF-WSE', 'EIxPF']
#     # name = 'branin'
#     # y_optimal = 8330
#     # BATCHS = len(functions) 
#     # RUNS = 10 # runs (of each algorithm)
#     # ITERS = 50 # iterations (on each run)
#     # NINITIAL = 10 # initial points (on each run)
#     # N = 2 # test every nth iters
#     # DATA = torch.ones(BATCHS, RUNS, ITERS)

#     # # filepath 
#     # if not os.path.exists('data'): 
#     #     os.makedirs('data') 
#     # file_path = os.path.join('data', 'branin.pt') 

#     # # # scripts 
#     # # for j, fun in enumerate(functions): 
#     # #     tic() 
#     # #     scriptloader(j, fun, DATA)  
#     # #     torch.save(DATA, file_path) 
#     # #     toc() 

#     # # load data  
#     # DATA = torch.load(file_path) 
#     # plotfig(DATA) 
 
#     # ======================================== #  
#     # prestress   
#     # ======================================== #  
#     functions = [bayesopt_rs, bayesopt_er, bayesopt_ei] 
#     names = ['RS', 'EIxPF-WSE', 'EIxPF'] 
#     name = 'prestressed' 
#     y_optimal = -17886 
#     BATCHS = len(functions) 
#     RUNS = 2 # runs (of each algorithm) 
#     ITERS = 40 # iterations (on each run) 
#     NINITIAL = 10 # initial points (on each run) 
#     N = 10 # test every nth iter 
#     DATA = torch.ones(BATCHS, RUNS, ITERS) 

#     # filepath 
#     if not os.path.exists('data'): 
#         os.makedirs('data') 
#     file_path = os.path.join('data', 'prestress.pt') 

#     # scripts 
#     for j, fun in enumerate(functions): 
#         tic() 
#         scriptloader(j, fun, DATA) 
#         torch.save(DATA, file_path) 
#         toc() 
 
#     # load data     
#     DATA = torch.load(file_path) 
#     plotfig(DATA) 

# type: ignore
import torch
import matplotlib.pyplot as plt
import os
from borc2.utilities import tic, toc 
import numpy as np
import json
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", message=r".*You are using*")  
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

from prestress_rs import bayesopt as bayesopt_rs
from prestress_ei import bayesopt as bayesopt_ei
from prestress_er import bayesopt as bayesopt_er

# Author: James Whiteley (github.com/jamesalexwhiteley)

def save_progress(progress_file, batch_n, run_n):
    """Save progress to a JSON file"""
    progress = {'batch': batch_n, 'run': run_n, 'timestamp': str(datetime.now())}
    with open(progress_file, 'w') as f:
        json.dump(progress, f)

def load_progress(progress_file):
    """Load progress from JSON file"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'batch': 0, 'run': 0}

def plotdata(d, x, data):
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'{name}_bayesopt.png')
    
    data = data[:, ::N]
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
        label_run = 'Sample runs' if i == 0 else None  # only label first 
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

def scriptloader(batch_n, fun, DATA, progress_file=None, start_run=0):
    """Modified to support resuming from a specific run"""
    for i in range(start_run, RUNS):
        print(f"Batch No. {batch_n} ({names[batch_n]}), Run No. {i}")
        try:
            # Add memory cleanup
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            _, data = fun(NINITIAL, ITERS, N) 
            
            # # Debug print to check data shape
            # if i == start_run:  # Only print once per batch
            #     print(f"  Data shape returned: {data.shape}")
            
            DATA[batch_n, i, :] = data
            
            # Save progress if progress_file provided
            if progress_file:
                save_progress(progress_file, batch_n, i + 1)
            
            # Save after each run to avoid losing progress
            torch.save(DATA, file_path)
            print(f"Completed and saved")
                
        except Exception as e:
            print(f"Error occurred in run {i} with function {fun.__name__}: {e}")
            import traceback
            traceback.print_exc()
            
            # Use previous run's data as fallback
            if i > 0:
                DATA[batch_n, i, :] = DATA[batch_n, i-1, :]
            else:
                DATA[batch_n, i, :] = torch.zeros(ITERS)
            
            # Save even on error
            if progress_file:
                save_progress(progress_file, batch_n, i + 1)
            torch.save(DATA, file_path)

if __name__ == "__main__":
    # ======================================== #  
    # prestress   
    # ======================================== #  
    functions = [bayesopt_rs, bayesopt_er, bayesopt_ei] 
    names = ['RS', 'EIxPF-WSE', 'EIxPF'] 
    name = 'prestressed' 
    y_optimal = -17886 
    BATCHS = len(functions) 
    RUNS = 2  # runs (of each algorithm) 
    ITERS = 50  # iterations (on each run) 
    NINITIAL = 5  # initial points (on each run) 
    N = 10  # test every nth iter 
    
    # filepath 
    if not os.path.exists('data'): 
        os.makedirs('data') 
    file_path = os.path.join('data', 'prestress.pt')
    progress_file = os.path.join('data', 'prestress_progress.json')
    
    # Resume functionality
    use_resume = True 
    
    if use_resume:
        # Load or initialize DATA and progress
        if os.path.exists(file_path):
            print("Loading existing data...")
            DATA = torch.load(file_path)
            progress = load_progress(progress_file)
            start_batch = progress['batch']
            start_run = progress['run']
            print(f"Resuming from batch {start_batch}, run {start_run}")
        else:
            print("Starting fresh...")
            DATA = torch.ones(BATCHS, RUNS, ITERS)
            start_batch = 0
            start_run = 0
        
        # scripts with resume
        for j in range(start_batch, len(functions)):
            fun = functions[j]
            print(f"\n{'='*50}")
            print(f"Starting {names[j]} (batch {j}/{len(functions)-1})")
            print(f"{'='*50}")
            
            tic()
            
            # If resuming mid-batch, start from saved run
            if j == start_batch:
                scriptloader(j, fun, DATA, progress_file, start_run)
            else:
                scriptloader(j, fun, DATA, progress_file, 0)
            
            # After completing a batch, update progress for next batch
            if j < len(functions) - 1:
                save_progress(progress_file, j + 1, 0)
            
            torch.save(DATA, file_path)
            toc()
        
        # Clean up progress file after successful completion
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("\nRemoved progress file (all done)")
    
    else:
        # Original behavior without resume
        DATA = torch.ones(BATCHS, RUNS, ITERS)
        
        # scripts 
        for j, fun in enumerate(functions): 
            tic() 
            scriptloader(j, fun, DATA) 
            torch.save(DATA, file_path) 
            toc() 
    
    # load data     
    DATA = torch.load(file_path) 
    plotfig(DATA)