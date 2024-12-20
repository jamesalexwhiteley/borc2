import torch 
import os 
import numpy as np 
from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from borc2.problem import Problem 
from borc2.surrogate import Surrogate
from borc2.acquisition import Acquisition
from borc2.bayesopt import Borc
from borc2.probability import  MultivariateNormal
from borc2.utilities import tic, toc 

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Author: James Whiteley (github.com/jamesalexwhiteley)

def plotfig(model, problem, borc, points0, points1, point_x, point_xi):

    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # domain 
    steps = 100
    nlevels = 10
    x = torch.linspace(0, 2, steps).to(device)
    xi = torch.linspace(-3, 3, steps).to(device)
    X, XI = torch.meshgrid(x, xi, indexing='ij')
    points = torch.column_stack([X.ravel(), XI.ravel()])

    # # true f(x,xi)
    # fig = plt.figure(figsize=(7, 6))
    # model(points)
    # z = model.f()
    # Z = z.view(X.shape)
    # contour = plt.contourf(X.cpu().numpy(), XI.cpu().numpy(), Z.cpu().numpy(), levels=np.linspace(Z.cpu().min(), Z.cpu().max(), nlevels), cmap='PuBu') 
    # plt.xlabel(r'$x$')
    # plt.ylabel(r'$\xi$')
    # cbar = plt.colorbar(contour, shrink=0.8, pad=0.05)
    # cbar.formatter = FormatStrFormatter('%.1f') 
    # cbar.update_ticks()
    # proxy_patch = Rectangle((0, 0), 1, 1, fc=plt.cm.PuBu(0.6),  edgecolor="none")
    # plt.legend([proxy_patch], [r'$f\,(x,\xi)$'], loc='upper right', fontsize=10, frameon=True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, f'analytic_f_0.png'), dpi=600)
    # plt.show()

    # # gp mean prediction f(x,xi)
    # fig = plt.figure(figsize=(7, 6))
    # gp = borc.surrogate.objective_gps[0]
    # mu = gp.predict(points).mu
    # MU = mu.view(X.shape)
    # contour = plt.contourf(X.cpu().numpy(), XI.cpu().numpy(), MU.cpu().numpy(), levels=np.linspace(Z.cpu().min(), Z.cpu().max(), nlevels), cmap='PuBu')
    # plt.scatter(points0[:, 0].cpu(), points0[:, 1].cpu(), s=75, marker='.', color='white', edgecolors='black', label='Initial Points')
    # plt.scatter(points1[:, 0].cpu(), points1[:, 1].cpu(), s=75, marker='.', color='red',   edgecolors='black', label='BayesOpt Points')
    # plt.xlabel(r'$x$')
    # plt.ylabel(r'$\xi$')
    # cbar = plt.colorbar(contour, shrink=0.8, pad=0.05)
    # cbar.formatter = FormatStrFormatter('%.1f')     
    # cbar.update_ticks()
    # plt.tight_layout()
    # plt.legend(loc='best')
    # plt.savefig(os.path.join(output_dir, f'analytic_fhat_0.png'), dpi=600)
    # plt.show()

    # # E[f(x,xi)]
    # fig, ax1 = plt.subplots(figsize=(7, 6))
    # # mean 
    # mu_true, prob_true = problem.rbo(x.unsqueeze(1), nsamples=int(1e4), output=False, return_vals=True)
    # opt_ind = torch.where(prob_true[0] > 0.95)[0][0]
    # opt_true_x, opt_true_mu, opt_true_prob = x[opt_ind].cpu(), mu_true[opt_ind].cpu(), prob_true[0][opt_ind].cpu()
    # ax1.scatter(opt_true_x, opt_true_mu, label='Optimal Point', color='white', marker='*', s=100, edgecolor='blue', zorder=10)
    # ax1.axvline(x=opt_true_x, color='gray', linestyle='--', linewidth=1)
    # ax1.plot(x.cpu(), mu_true.cpu(), label=r'True $\text{E}[f(x,\xi)]$', color='blue', linewidth=2)
    # ax1.set_xlabel(r'$x$')
    # ax1.set_ylabel(r'$\text{E}[f(x,\xi)]$', color='blue')
    # ax1.tick_params(axis='y', labelcolor='blue')
    # mu_gp, mu2_gp, prob_gp, prob2_gp = borc.rbo(x.unsqueeze(1), nsamples=int(1e4), output=False, return_posterior=True)
    # ax1.plot(x.cpu(), mu_gp.cpu(), label=r'GP Mean $\text{E}[f(x,\xi)]$', color='green', linestyle='--', linewidth=2)
    # upper_bound, lower_bound = mu_gp.cpu() + 2 * mu2_gp.cpu(), mu_gp.cpu() - 2 * mu2_gp.cpu()
    # ax1.fill_between(x.cpu(), lower_bound, upper_bound, color='green', alpha=0.1)
    # # ax1.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    # # probability 
    # ax2 = ax1.twinx()
    # ax2.scatter(opt_true_x, opt_true_prob, label=r'$\text{P}[g(x,\xi)\leq0]=0.95$', color='white', marker='o', s=50, edgecolor='red', zorder=10)
    # ax2.axhline(y=opt_true_prob, color='gray', linestyle='--', linewidth=1)
    # ax2.plot(x.cpu(), prob_true[0].cpu(), label=r'True $\text{P}[g(x,\xi)\leq0]$', color='red', linewidth=2)
    # ax2.plot(x.cpu(), prob_gp[0].cpu(), label=r'GP Mean $\text{P}[g(x,\xi)\leq0]$', color='orange', linestyle='--', linewidth=2)
    # # upper_bound, lower_bound = prob_gp[0].cpu() + 2 * prob2_gp[0].cpu(), prob_gp[0].cpu() - 2 * prob2_gp[0].cpu()
    # # ax2.fill_between(x.cpu(), lower_bound, upper_bound, color='red', alpha=0.1)
    # ax2.set_ylabel(r'$\text{P}[g(x,\xi)\leq0]$', color='red')
    # ax2.tick_params(axis='y', labelcolor='red')
    # # # acquisition
    # # ax3 = ax1.twinx() 
    # # borc.acquisition = Acquisition(f="eMU", g="ePF", xi=problem.sample_xi(nsamples=int(1e4)).to(device), eps=0.05) 
    # # acq = borc.eval_acquisition(x.unsqueeze(1)).detach()
    # # ax3.plot(x.cpu(), acq.cpu(), label=r'Acquisition function$', color='gray', linewidth=1) 
    # # ax3.get_yaxis().set_visible(False)
    # # legend 
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # fig.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.15, 0.95))
    # fig.tight_layout()
    # plt.savefig(os.path.join(output_dir, f'analytic_posterior_0.png'), dpi=600)
    # plt.show()

    # acquisition 
    fig, ax1 = plt.subplots(figsize=(7, 6.5))
    borc.acquisition = Acquisition(f="eMU", g="ePF", xi=problem.sample_xi(nsamples=int(1e4)).to(device), eps=0.05)
    acq_x = borc.eval_acquisition(x.unsqueeze(1)).detach()
    ax1.plot(x.cpu(), acq_x.cpu(), label=r'Acquisition function', color='#1f3d7a', linewidth=2)
    ax1.set_xlabel(r'$x$', fontsize=12)
    ax1.set_ylabel(r'$\alpha(x)$', fontsize=12)
    ax1.scatter(point_x[0].cpu(), borc.eval_acquisition(point_x[0].unsqueeze(0)).detach().cpu(), color='m', s=25, marker='o', label='Max acquisition', zorder=10) 
    ax1.axvline(point_x[0].cpu(), color='gray', linestyle='--', linewidth=1) 
    ax1.legend(loc='lower right', frameon=True)
    # xi insert 
    ax_inset = inset_axes(ax1, width="40%", height="40%", loc='upper left') 
    borc.acquisition = Acquisition(f="eWMSE", x=torch.tensor([1.0]).to(device), dist=problem.param_dist)
    acq_xi = borc.eval_acquisition(xi.unsqueeze(1)).detach()
    ax_inset.plot(xi.cpu(), acq_xi.cpu(), label=r'Acquisition $\alpha(\xi)$', color='#1f3d7a', linewidth=2)
    ax_inset.set_xlabel(r'$\xi$', fontsize=10)
    ax_inset.set_ylabel(r'$\alpha(\xi)$', fontsize=10)
    ax_inset.tick_params(axis='both', labelsize=8)
    ax_inset.yaxis.tick_right()
    ax_inset.yaxis.set_label_position("right")
    ax_inset.scatter(point_xi[0].cpu(), borc.eval_acquisition(point_xi[0].unsqueeze(0)).detach().cpu(), color='m', marker='.', label='Max acquisition', zorder=10) 
    ax_inset.axvline(point_xi[0].cpu(), color='gray', linestyle='--', linewidth=1) 
    plt.savefig(os.path.join(output_dir, f'analytic_acquisition_0.png'), dpi=600)
    plt.show()

class Model():
    def __call__(self, x): 
        self.x = x
        self.m = None

    def f(self):
        x, xi = self.x[:, 0], self.x[:, 1]
        return torch.sin(torch.pi * x) / (1 + torch.exp(-10 * (xi - 1/2))) + x**2 - xi**2 + 10
    
    def g(self):
        x, xi = self.x[:, 0], self.x[:, 1]
        # return xi - 1/2 * x**2 - 1
        # return -torch.cos(x) / (1 + xi**2) + x * xi - x**3 - 1
        return -x**2 * torch.sin(xi) - x**3 + xi**3 - torch.log(1 + xi**2) - 1
        
def bayesopt(ninitial, iters):

    problem = Problem()
    model = Model()
    bounds = {"x": (0, 2)}
    mu = torch.tensor([0.]) 
    cov = torch.tensor([[1.]])
    dist = MultivariateNormal(mu, cov)

    problem.set_bounds(bounds)
    problem.set_dist(dist)
    problem.add_model(model)
    problem.add_objectives([model.f])
    problem.add_constraints([model.g])

    xi = problem.sample_xi(nsamples=int(1e4)).to(device)
    surrogate = Surrogate(problem, ntraining=100, nstarts=5) 
    acquisition = Acquisition(f="eMU", g="ePF", xi=xi, eps=0.1) 
    borc = Borc(surrogate, acquisition) 
    borc.cuda(device) 
    borc.initialize(nsamples=ninitial, sample_method="sobol", max_acq=torch.tensor([0.0])) 

    points0, _ = borc.surrogate.objective_gps[0].get_training_data()
    # points1 = [torch.tensor([[0.5, -1]]), torch.tensor([[1.5, 0.75]])]
    points1 = []

    for _ in range(iters):

        # borc.acquisition = Acquisition(f="eMU", xi=xi)  
        # _, borc.fbest = borc.batch_optimize_acq(iters=50, nstarts=5, optimize_x=True) 

        borc.acquisition = Acquisition(f="PI") 
        new_x, max_acq_x = borc.batch_optimize_acq(iters=2, nstarts=1) 

        # # borc.acquisition = Acquisition(f="eEI", g="ePF", xi=xi) 
        # borc.acquisition = Acquisition(f="eEI", xi=xi) 
        # new_x, max_acq_x = borc.batch_optimize_acq(iters=10, nstarts=1, optimize_x=True) 
 
        # borc.acquisition = Acquisition(f="eWMSE", x=new_x, dist=problem.param_dist)  
        # new_xi, max_acq_xi = borc.batch_optimize_acq(iters=100, nstarts=5, optimize_xi=True)
        # borc.step(new_x=torch.cat([new_x, new_xi], dim=1))
        # points1.append(torch.cat([new_x, new_xi], dim=1))
        # print(f"new_x : {torch.cat([new_x, new_xi], dim=1)}") 

        # borc.acquisition = Acquisition(f="eMU", xi=xi) 
        # _, borc.fbest = borc.batch_optimize_acq(iters=50, nstarts=5, optimize_x=True) 

        # plotfig(model, problem, borc, points0, torch.cat(points1), torch.tensor((new_x, max_acq_x)), torch.tensor((new_xi, max_acq_xi)))

    # plotfig(model, problem, borc, points0, torch.cat(points1), (new_x, max_acq_x), (new_xi, max_acq_xi))

    return None, None 


if __name__ == "__main__": 
    ninitial, iters = 75, 1 
    xopt, res = bayesopt(ninitial, iters) 

    # TODO extreme z input to self.normal.cdf(z) is breaking gradient flow 
    # TODO possibly incorrect fbest (ok for determinsitic optimisation) is the cause of this (maybe not)? 
    # TODO why is std so small, I wonder? 