# type: ignore
import torch
import matplotlib.pyplot as plt
import os
from borc2.utilities import tic, toc 
import numpy as np
import json
from datetime import datetime
import gc

import warnings
warnings.filterwarnings("ignore", message=r".*You are using*")  
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

from prestress_rs import bayesopt as bayesopt_rs
from prestress_ei import bayesopt as bayesopt_ei
from prestress_er import bayesopt as bayesopt_er

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

def plotdata(d, x, data, names, name, y_optimal):
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'{name}_bayesopt_{names[d]}.png')
    
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
        label_run = 'Sample runs' if i == 0 else None
        plt.plot(x, individual_error, '-', color=colours[d], 
                alpha=show_alpha, linewidth=1, label=label_run)

    # Percentile bands
    p25 = torch.quantile(error, 0.25, dim=0).numpy()
    p50 = torch.quantile(error, 0.50, dim=0).numpy()
    p75 = torch.quantile(error, 0.75, dim=0).numpy()
    plt.fill_between(x, p25, p75, color=colours[d], alpha=0.2, label=f'{label} (IQR)')
    plt.plot(x, p50, color=colours[d], linewidth=2.5, alpha=1.0, label=f'{label} (median)')
    
    # Markers
    plt.scatter(x, p50, color=colours[d], s=30, edgecolor=colours[d])

    # Background lines 
    plt.axhline(y=0.0, color='#666666', linestyle='-', linewidth=1, alpha=0.7)
    plt.grid(True, linestyle='-', alpha=0.2, color='gray')

    # Labels 
    plt.xlabel("Iterations", fontsize=fontsize)
    plt.ylabel(r"Absolute Error $\quad$ $|y^* - \mathbb{E}[f(x^*, \xi)]|$", fontsize=fontsize)
    plt.legend(fontsize=fontsize-2, loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.show()

def plotfig(DATA, names, name, y_optimal): 
    print(f"\nPlotting results...")
    print(f"Data shape: {DATA.shape}")
    x = torch.arange(DATA.size(2))
    for d, data in enumerate(DATA):
        plotdata(d, x, data, names, name, y_optimal)

def scriptloader(batch_n, fun, DATA, progress_file, start_run, params):
    """Run experiments for one function"""
    for i in range(start_run, params['RUNS']):
        print(f"  Run {i}/{params['RUNS']-1}...")
        try:
            # Clean up memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            # Set seeds for reproducibility
            torch.manual_seed(42 + batch_n * 100 + i)
            np.random.seed(42 + batch_n * 100 + i)
            
            # Run the bayesopt function
            _, data = fun(params['NINITIAL'], params['ITERS'], params['N'])
            
            # Store results
            DATA[batch_n, i, :] = data
            
            # Save progress
            save_progress(progress_file, batch_n, i + 1)
            torch.save(DATA, params['file_path'])
            print(f"    Completed")
                
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fill with NaN on error
            DATA[batch_n, i, :] = torch.full((params['EXPECTED_RESULTS'],), float('nan'))
            
            # Save even on error
            save_progress(progress_file, batch_n, i + 1)
            torch.save(DATA, params['file_path'])

def main():
    # Configuration
    functions = [bayesopt_rs, bayesopt_er, bayesopt_ei] 
    names = ['RS', 'EIxPF-WSE', 'EIxPF'] 
    name = 'prestressed' 
    y_optimal = -17886 
    
    # Experiment parameters
    RUNS = 1        # runs of each algorithm
    ITERS = 6      # iterations per run
    NINITIAL = 10    # initial points per run
    N = 3           # test every Nth iteration
    
    # Derived values
    BATCHS = len(functions)
    EXPECTED_RESULTS = ITERS
    
    print(f"Configuration:")
    print(f"  Functions: {len(functions)}")
    print(f"  Runs per function: {RUNS}")
    print(f"  Iterations per run: {ITERS}")
    print(f"  Expected tensor size: {EXPECTED_RESULTS}")
    
    # File paths
    os.makedirs('data', exist_ok=True)
    file_path = os.path.join('data', 'prestress.pt')
    progress_file = os.path.join('data', 'prestress_progress.json')
    
    # Load or create data
    if os.path.exists(file_path):
        print(f"\nLoading existing data from {file_path}")
        DATA = torch.load(file_path)
        
        # Check shape
        if DATA.shape == (BATCHS, RUNS, EXPECTED_RESULTS):
            print(f"  Data shape correct: {DATA.shape}")
            progress = load_progress(progress_file)
            start_batch = progress['batch']
            start_run = progress['run']
            if start_batch < BATCHS:
                print(f"  Resuming from batch {start_batch}, run {start_run}")
            else:
                print(f"  All batches complete, will plot results")
        else:
            # print(f"  WARNING: Data shape {DATA.shape} doesn't match expected {(BATCHS, RUNS, EXPECTED_RESULTS)}")
            print(f"  Starting fresh...")
            DATA = torch.ones(BATCHS, RUNS, EXPECTED_RESULTS) * float('nan')
            start_batch = 0
            start_run = 0
    else:
        print(f"\nNo existing data found, starting fresh...")
        DATA = torch.ones(BATCHS, RUNS, EXPECTED_RESULTS) * float('nan')
        start_batch = 0
        start_run = 0
    
    # Parameters dictionary for cleaner passing
    params = {
        'RUNS': RUNS,
        'ITERS': ITERS,
        'NINITIAL': NINITIAL,
        'N': N,
        'EXPECTED_RESULTS': EXPECTED_RESULTS,
        'file_path': file_path
    }
    
    # Run experiments
    for j in range(start_batch, BATCHS):
        print(f"\n{'='*50}")
        print(f"Running {names[j]} (batch {j}/{BATCHS-1})")
        print(f"{'='*50}")
        
        tic()
        
        # Determine starting run
        if j == start_batch:
            scriptloader(j, functions[j], DATA, progress_file, start_run, params)
        else:
            scriptloader(j, functions[j], DATA, progress_file, 0, params)
        
        # Update progress for next batch
        if j < BATCHS - 1:
            save_progress(progress_file, j + 1, 0)
        
        toc()
    
    # Clean up and plot
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print("\nAll experiments complete, removed progress file")
    
    # Final save
    torch.save(DATA, file_path)
    
    # Plot results
    plotfig(DATA, names, name, y_optimal)

if __name__ == "__main__":
    main()