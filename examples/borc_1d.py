import torch 
from matplotlib import pyplot as plt
import matplotlib

# from borc2.surrogate import GPSurrogate
# from borc2.problem import Problem 
# from borc2.acquisition import Acquisition
# from borc2.BORC import BORC
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = ['Verdana']

# # Author: James Whiteley (github.com/jamesalexwhiteley)

# def plot1d(problem, borc):

#     fig = plt.figure(figsize=(8, 6))
#     fig.add_subplot(2, 1, 1)

#     # underlying function 
#     x = torch.linspace(problem.param_bounds["x"][0], problem.param_bounds["x"][1], 1000).unsqueeze(1)
#     y = (model(x), model.f())[1]
#     plt.plot(x, y, label='Objective function', color='k', linewidth=1)
#     plt.ylabel('f(x)') 

#     # gp
#     x = x.squeeze(1)
#     pred = borc.surrogate.predict_objectives(x.unsqueeze(-1).to(device), return_std=True, grad=False)[0]
#     pred.cuda('cpu')
#     mu, std = pred.mu, pred.std
#     low, high = mu - 2 * std, mu + 2 * std 
#     plt.plot(x, mu, label='GP posterior mean', color='b')
#     plt.fill_between(x, low, high, where=(high > low), interpolate=True, color='b', alpha=0.15, label=r'GP posterior 95% bounds')
#     # sampled points 
#     gp = borc.surrogate.objective_gps[0]
#     train_x, train_y = gp.get_training_data(device='cpu')
#     plt.scatter(train_x.flatten(), train_y.flatten(), color='k', label='{} training points'.format(len(train_x)), marker='o', s=35)
#     plt.legend(loc=0)
    
#     # acquisition function 
#     fig.add_subplot(2, 1, 2)
#     pred.cuda('cuda')
#     a = borc.eval_acquisition(x.unsqueeze(1).to(device)).detach()
#     plt.plot(x, a.to('cpu'), label='Acquisition function', color='k')
#     plt.scatter(new_x.to('cpu'), max_acq.to('cpu'), label='Max acquisition', color='m', s=30, marker='D')
#     plt.xlabel('x')
#     plt.legend(loc=0)
#     plt.ylabel(r'$\alpha$(x)') 
#     # plt.savefig('borc_1d.png', dpi=400)
#     plt.show()

# class Model():
#     def __call__(self, x):
#         self.m = x**2 + 3 * torch.sin(3 * x)

#     def f(self):
#         return self.m.flatten() 

# if __name__ == "__main__": 

#     problem = Problem()
#     model = Model()
#     bounds = {"x": (-3, 3)}
#     problem.set_bounds(bounds)
#     problem.add_model(model)
#     problem.add_objectives([model.f])
    
#     surrogate = GPSurrogate()
#     acquisition = Acquisition(f="EI")
#     borc = BORC(problem, surrogate, acquisition) 
#     borc.cuda(device)
#     borc.initialize(nsamples=6, sample_method="lhs") 

#     iters = 3
#     for i in range(iters): 
#         print(f"Iter: {i + 1}/{iters} | Max Objective: {borc.fbest},  Optimal x : {borc.xbest}") 
#         new_x, max_acq = borc.optimize_acq(iters=50, nstarts=20, batch_mode=True) # use batch mode with cuda 
#         plot1d(problem, borc) 
#         borc.step(new_x=new_x) 
       
