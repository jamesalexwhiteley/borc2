import torch 
import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec

from borc2.problem import Problem 
from borc2.surrogate import Surrogate
from borc2.acquisition import Acquisition
from borc2.bayesopt import Borc
from borc2.probability import  MultivariateNormal
from borc2.utilities import tic, toc 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Author: James Whiteley (github.com/jamesalexwhiteley)

def plotfig(model, problem, borc, initial_points, new_points):

    # domain 
    steps = 100
    nlevels = 10
    x = torch.linspace(0, 2, steps).cuda(device)
    xi = torch.linspace(-1, 1, steps).cuda(device)
    X, XI = torch.meshgrid(x, xi, indexing='ij')
    points = torch.column_stack([X.ravel(), XI.ravel()])

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # true f(x,xi)
    model(points)
    z = model.f()
    Z = z.view(X.shape)
    contour = axs[0, 0].contourf(X.cpu().numpy(), XI.cpu().numpy(), Z.cpu().numpy(), levels=np.linspace(Z.cpu().min(), Z.cpu().max(), nlevels), cmap='PuBu') 
    axs[0, 0].set_xlabel(r'$x$')
    axs[0, 0].set_ylabel(r'$\xi$')
    cbar = fig.colorbar(contour, ax=axs[0, 0], shrink=0.8, pad=0.05)
    cbar.formatter = FormatStrFormatter('%.1f') 
    cbar.update_ticks()

    # gp mean prediction f(x,xi)
    gp = borc.surrogate.objective_gps[0]
    mu = gp.predict(points).mu
    MU = mu.view(X.shape)
    contour = axs[0, 1].contourf(X.cpu().numpy(), XI.cpu().numpy(), MU.cpu().numpy(), levels=np.linspace(Z.cpu().min(), Z.cpu().max(), nlevels), cmap='PuBu')
    axs[0, 1].set_xlabel(r'$x$')
    axs[0, 1].set_ylabel(r'$\xi$')
    cbar = fig.colorbar(contour, ax=axs[0, 1], shrink=0.8, pad=0.05)
    cbar.formatter = FormatStrFormatter('%.1f')  
    cbar.update_ticks()

    # true E[f(x,xi)]
    mu, prob = problem.rbo(x.unsqueeze(1), nsamples=int(1e4), output=False, return_vals=True) 
    axs[1, 0].plot(x.cpu(), mu.cpu())
    # axs[1, 0].plot(x.cpu(), prob[0].cpu())
    # gp mean prediction E[f(x,xi)]
    mu, prob = borc.rbo(x.unsqueeze(1), nsamples=int(1e4), output=False, return_vals=True) 
    axs[1, 0].plot(x.cpu(), mu.cpu())
    # axs[1, 0].plot(x.cpu(), prob[0].cpu())

    # # objective F(x) 
    # # f = -x**2 
    # # axs[1, 1].plot(x.cpu(), f.cpu(), label=r'$-x^2$', color='k')
    # # mu_f = gp.posterior_mean_at_xi(x.unsqueeze(1), xi=borc.problem.get_xi_mean_vector().to(device)).detach()
    # axs[1, 1].plot(x.cpu(), mu_f.cpu(), '--', label=r'$a_n(x)$', color='b')
    # axs[1, 1].set_xlabel("x")
    # axs[1, 1].set_ylabel("E[f(x, xi)]")

    # # acquisition function 
    # acq = borc.eval_acquisition(points).detach()
    # ACQ = acq.view(X.shape)
    # contour = axs[1, 0].contourf(X.cpu().numpy(), XI.cpu().numpy(), ACQ.cpu().numpy(), cmap='inferno')
    # # axs[1, 0].scatter(initial_points[:, 0].cpu(), initial_points[:, 1].cpu(), color='k', marker='*')
    # # if new_points != []:
    # #     axs[1, 0].scatter(torch.cat(new_points, dim=0)[:, 0].cpu(), torch.cat(new_points, dim=0)[:, 1].cpu(), color='g', marker='*')
    # axs[1, 0].set_xlabel('x')
    # axs[1, 0].set_ylabel('xi')
    # fig.colorbar(contour, ax=axs[1, 0])

    # for ax in fig.get_axes():
    #     ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # plt.legend()
    plt.tight_layout()
    plt.show()

class Model():
    def __call__(self, x): 
        self.x = x
        self.m = None

    def f(self):
        x, xi = self.x[:, 0], self.x[:, 1]
        return torch.sin(5 * torch.pi * x) / (1 + torch.exp(-5 * (xi - 0.5))) + x**2 - xi**2 + 1
        # return torch.tanh(torch.pi * x) * torch.tanh(5/3 * torch.pi * xi) + x**2 - xi**2 + 1
    
    def g(self):
        x, xi = self.x[:, 0], self.x[:, 1]
        return xi - 0.5 * x**2 - 1
        
def bayesopt(ninitial, iters):

    problem = Problem()
    model = Model()
    bounds = {"x": (0, 2)}
    mu = torch.tensor([0.0]) 
    cov = torch.tensor([[1.0]])
    dist = MultivariateNormal(mu, cov)

    problem.set_bounds(bounds)
    problem.set_dist(dist)
    problem.add_model(model)
    problem.add_objectives([model.f])
    problem.add_constraints([model.g])

    xi = problem.sample_xi(nsamples=int(2e2)).to(device)
    surrogate = Surrogate(problem, ntraining=100, nstarts=5) 
    acquisition = Acquisition(f="eMU", g="ePF", xi=xi, eps=0.1) 
    borc = Borc(surrogate, acquisition) 
    borc.cuda(device) 
    borc.initialize(nsamples=ninitial, sample_method="sobol", max_acq=torch.tensor([0])) 

    initial_points, _ = borc.surrogate.objective_gps[0].get_training_data()
    new_points = []

    # for _ in range(iters):

    #     tic()
    #     new_x, max_acq = borc.batch_optimize_acq(iters=50, nstarts=10, optimize_x=True)   
    #     new_points.append(new_x)
    #     print(f"new_x : {new_x} | max_acq : {max_acq}") 
    #     toc()
    #     plotfig(model, borc, initial_points, new_points) 
    #     borc.step(new_x=new_x) 

    plotfig(model, problem, borc, initial_points, new_points)

    return None, None 


if __name__ == "__main__": 
    ninitial, iters = 200, 10 
    xopt, res = bayesopt(ninitial, iters) 