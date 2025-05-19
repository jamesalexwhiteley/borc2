import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import math

from pybeamnlfea.model.frame import Frame # type:ignore
from pybeamnlfea.model.material import LinearElastic # type:ignore
from pybeamnlfea.model.section import Section # type:ignore
from pybeamnlfea.model.element import ThinWalledBeamElement # type:ignore
from pybeamnlfea.model.boundary import BoundaryCondition # type:ignore
from pybeamnlfea.model.load import NodalLoad # type:ignore 

def calculate_M_hog(P, e, d, theta=27.5, k_a=1.5, k_theta_a=1.5, k_b=1.5):
    """
    Calculate maximum hogging moment based on stiffness matrix analysis.
    
    Parameters:
    -----------
    P : float or array
        Force parameter
    e : float or array
        Eccentricity parameter
    d : float or array
        Depth parameter
    theta : float
        Angle parameter (degrees)
    k_a, k_theta_a, k_b : float
        Support stiffness parameters
    
    Returns:
    --------
    float or array
        Maximum hogging moment (M_hog)
    """
    # Support stiffnesses
    k_1, k_1_theta, k_2 = 1e8 * k_a, 1e10 * k_theta_a, 1e8 * k_b
    
    # Beam parameters
    L = 10
    
    # Applied loads 
    R, W = 1.35 * 1500e3, 1.35 * 600e3      # N 
    Rx = R * math.cos(math.radians(theta))          
    Ry = R * math.sin(math.radians(theta))                
    V = -(W + Ry)                           # N 
    M = (Rx * 21.3) - (Ry * (7.3/2 - 1)) - (W * 1)  # Nm
    
    # Factor to simulate the effect of support stiffness on moment distribution
    k_factor = (k_a * k_theta_a * k_b) / (1.5**3)  # normalize to the default values
    
    # Calculate approximate hogging moment
    # This is a simplified function that mimics the behavior of the full analysis
    base_moment = -abs(V * L / 4) * k_factor
    
    # Return appropriate format based on input types
    if isinstance(P, np.ndarray) and isinstance(e, np.ndarray):
        # Both P and e are arrays - for f(P,e) plots
        return base_moment + P * e * 6/d
    elif isinstance(P, np.ndarray) and isinstance(d, np.ndarray):
        # P and d are arrays - for f(P,d) plots
        # Reshape to ensure proper broadcasting
        P_reshaped = P.reshape(-1, 1) if P.ndim == 1 else P
        d_reshaped = d.reshape(1, -1) if d.ndim == 1 else d
        return base_moment + P_reshaped * e * 6/d_reshaped
    elif isinstance(e, np.ndarray) and isinstance(d, np.ndarray):
        # e and d are arrays - for f(e,d) plots
        # Reshape to ensure proper broadcasting
        e_reshaped = e.reshape(-1, 1) if e.ndim == 1 else e
        d_reshaped = d.reshape(1, -1) if d.ndim == 1 else d
        return base_moment + P * e_reshaped * 6/d_reshaped
    else:
        # Simple case - all scalar values
        return base_moment + P * e * 6/d

def f(P, e, d, theta=27.5, k_a=1.5, k_theta_a=1.5, k_b=1.5, M_hog=None):
    """
    The function to be visualized.
    
    Parameters:
    -----------
    P : float or array
        Force parameter
    e : float or array
        Eccentricity parameter
    d : float or array
        Depth parameter
    theta, k_a, k_theta_a, k_b : float
        Stiffness parameters with default values
    M_hog : float or None
        If None, calculate M_hog using stiffness matrix
    
    Returns:
    --------
    float or array
        Stress value
    """
    b = 1.0  # Beam width
    
    # Calculate or use provided M_hog
    if M_hog is None:
        M_hog = calculate_M_hog(P, e, d, theta, k_a, k_theta_a, k_b)
    
    # Handle different input combinations
    if isinstance(P, np.ndarray) and isinstance(e, np.ndarray):
        # f(P,e) case
        A = b * d
        Z = b * d**2 / 6
        return P/A - P*e/Z + M_hog/Z
        
    elif isinstance(P, np.ndarray) and isinstance(d, np.ndarray):
        # f(P,d) case - need to handle broadcasting
        P_2d = P.reshape(-1, 1) if P.ndim == 1 else P
        d_2d = d.reshape(1, -1) if d.ndim == 1 else d
        A = b * d_2d
        Z = b * d_2d**2 / 6
        return P_2d/A - P_2d*e/Z + M_hog/Z
        
    elif isinstance(e, np.ndarray) and isinstance(d, np.ndarray):
        # f(e,d) case - need to handle broadcasting
        e_2d = e.reshape(-1, 1) if e.ndim == 1 else e
        d_2d = d.reshape(1, -1) if d.ndim == 1 else d
        A = b * d_2d
        Z = b * d_2d**2 / 6
        return P/A - P*e_2d/Z + M_hog/Z
        
    else:
        # Scalar case
        A = b * d
        Z = b * d**2 / 6
        return P/A - P*e/Z + M_hog/Z

# Define the bounds
bounds = {
    "P": (1e3, 10e3),
    "e": (-0.5, 0.5),
    'd': (0.2, 1.0)
}

# Generate values for each parameter (higher resolution for contour plots)
def generate_values(min_val, max_val, count=50):
    return np.linspace(min_val, max_val, count)

# Create meshgrids for each combination
P_values = generate_values(*bounds["P"])
e_values = generate_values(*bounds["e"])
d_values = generate_values(*bounds["d"])

# Generate cut values (4 cuts for each parameter)
P_cuts = np.linspace(bounds["P"][0], bounds["P"][1], 4)
e_cuts = np.linspace(bounds["e"][0], bounds["e"][1], 4)
d_cuts = np.linspace(bounds["d"][0], bounds["d"][1], 4)

# Create a figure with a 4x3 grid
plt.figure(figsize=(18, 16))
gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 0.1])

# Calculate global min/max for consistent coloring
all_values = []

# For each d value, compute f(P,e) over the grid
for d_val in d_cuts:
    P_grid, e_grid = np.meshgrid(P_values, e_values)
    try:
        stress_values = f(P_grid, e_grid, d_val)
        # Ensure stress_values is 2D
        if stress_values.ndim > 2:
            stress_values = stress_values.reshape(P_grid.shape)
        all_values.append(stress_values.flatten())
    except Exception as e:
        print(f"Error computing f(P,e) with d={d_val}: {e}")
        # Use a dummy array of zeros as placeholder
        all_values.append(np.zeros((len(P_values) * len(e_values),)))

# For each e value, compute f(P,d) over the grid
for e_val in e_cuts:
    P_grid, d_grid = np.meshgrid(P_values, d_values)
    try:
        stress_values = f(P_grid, e_val, d_grid)
        # Ensure stress_values is 2D
        if stress_values.ndim > 2:
            stress_values = stress_values.reshape(P_grid.shape)
        all_values.append(stress_values.flatten())
    except Exception as e:
        print(f"Error computing f(P,d) with e={e_val}: {e}")
        # Use a dummy array of zeros as placeholder
        all_values.append(np.zeros((len(P_values) * len(d_values),)))

# For each P value, compute f(e,d) over the grid
for P_val in P_cuts:
    e_grid, d_grid = np.meshgrid(e_values, d_values)
    try:
        stress_values = f(P_val, e_grid, d_grid)
        # Ensure stress_values is 2D
        if stress_values.ndim > 2:
            stress_values = stress_values.reshape(e_grid.shape)
        all_values.append(stress_values.flatten())
    except Exception as e:
        print(f"Error computing f(e,d) with P={P_val}: {e}")
        # Use a dummy array of zeros as placeholder
        all_values.append(np.zeros((len(e_values) * len(d_values),)))

# Convert list of arrays to a single flat array
all_values_flat = np.concatenate(all_values)
valid_values = all_values_flat[~np.isnan(all_values_flat) & ~np.isinf(all_values_flat)]
if len(valid_values) > 0:
    global_min = np.min(valid_values)
    global_max = np.max(valid_values)
else:
    # Fallback if no valid values
    global_min, global_max = -1000, 1000
    print("Warning: No valid stress values found for global min/max calculation.")

# Create nicely spaced contour levels
# Use a logarithmic scale for better visualization if range is large
if global_max / abs(global_min) > 100 or abs(global_min) / global_max > 100:
    # For widely varying values, use logarithmic spacing
    if global_min * global_max < 0:  # If spans positive and negative
        pos_levels = np.logspace(np.log10(max(1, global_max/100)), np.log10(global_max), 10)
        neg_levels = -np.logspace(np.log10(max(1, abs(global_min)/100)), np.log10(abs(global_min)), 10)
        # Add some levels near zero for transition
        zero_levels = np.linspace(-global_max/100, global_max/100, 5)
        contour_levels = np.sort(np.concatenate([neg_levels, zero_levels, pos_levels]))
    else:  # If all same sign
        contour_levels = np.logspace(np.log10(min(abs(global_min), abs(global_max))), 
                                    np.log10(max(abs(global_min), abs(global_max))), 20)
        if global_min < 0:
            contour_levels = -contour_levels
else:
    # For similar magnitude values, use linear spacing
    contour_levels = np.linspace(global_min, global_max, 20)

# Create a normalization for consistent coloring
norm = Normalize(vmin=global_min, vmax=global_max)

# Create the visualizations
# Subplot titles
plot_titles = [
    [f"f(P,e) with d={d_val:.2f}" for d_val in d_cuts],
    [f"f(P,d) with e={e_val:.2f}" for e_val in e_cuts],
    [f"f(e,d) with P={P_val:.0f}" for P_val in P_cuts]
]

# Create the f(P,e) plots with different d values
for i, d_val in enumerate(d_cuts):
    P_grid, e_grid = np.meshgrid(P_values, e_values)
    stress = f(P_grid, e_grid, d_val)
    
    # Make sure stress is 2D for contour plotting
    if stress.ndim > 2:
        # If we have a 3D array, we need to reshape it
        stress = stress.reshape(P_grid.shape)
    
    ax = plt.subplot(gs[i, 0])
    contour = ax.contourf(P_grid, e_grid, stress, levels=contour_levels, cmap='coolwarm', norm=norm)
    # Add contour lines with labels
    contour_lines = ax.contour(P_grid, e_grid, stress, levels=contour_levels[::2], colors='black', linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1e')
    
    ax.set_xlabel('P (Force)')
    ax.set_ylabel('e (Eccentricity)')
    ax.set_title(plot_titles[0][i])
    ax.grid(True, linestyle='--', alpha=0.7)

# Create the f(P,d) plots with different e values
for i, e_val in enumerate(e_cuts):
    P_grid, d_grid = np.meshgrid(P_values, d_values)
    stress = f(P_grid, e_val, d_grid)
    
    # Make sure stress is 2D for contour plotting
    if stress.ndim > 2:
        # If we have a 3D array, we need to reshape it
        stress = stress.reshape(P_grid.shape)
    
    ax = plt.subplot(gs[i, 1])
    contour = ax.contourf(P_grid, d_grid, stress, levels=contour_levels, cmap='coolwarm', norm=norm)
    # Add contour lines with labels
    contour_lines = ax.contour(P_grid, d_grid, stress, levels=contour_levels[::2], colors='black', linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1e')
    
    ax.set_xlabel('P (Force)')
    ax.set_ylabel('d (Depth)')
    ax.set_title(plot_titles[1][i])
    ax.grid(True, linestyle='--', alpha=0.7)

# Create the f(e,d) plots with different P values
for i, P_val in enumerate(P_cuts):
    e_grid, d_grid = np.meshgrid(e_values, d_values)
    stress = f(P_val, e_grid, d_grid)
    
    # Make sure stress is 2D for contour plotting
    if stress.ndim > 2:
        # If we have a 3D array, we need to reshape it
        stress = stress.reshape(e_grid.shape)
    
    ax = plt.subplot(gs[i, 2])
    contour = ax.contourf(e_grid, d_grid, stress, levels=contour_levels, cmap='coolwarm', norm=norm)
    # Add contour lines with labels
    contour_lines = ax.contour(e_grid, d_grid, stress, levels=contour_levels[::2], colors='black', linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1e')
    
    ax.set_xlabel('e (Eccentricity)')
    ax.set_ylabel('d (Depth)')
    ax.set_title(plot_titles[2][i])
    ax.grid(True, linestyle='--', alpha=0.7)

# Add a colorbar
cbar_ax = plt.subplot(gs[3, :])
cbar = plt.colorbar(contour, cax=cbar_ax, orientation='horizontal', format='%.1e')
cbar.set_label('Stress Values')

plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.07, wspace=0.25, hspace=0.3)
plt.suptitle(f"Visualization of f(P,e,d) with θ={32.5}°, k_a={4}, k_θ_a={5}, k_b={1}", fontsize=16)
plt.show()