import numpy as np 
from matplotlib import pyplot as plt 
from scipy.stats import lognorm

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import lognorm

plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Roboto']

shape = 0.5
scale = np.exp(0)
x = np.linspace(0, 5, 1000)
x_fill = np.linspace(0, 0.5, 1000)
pdf = lognorm.pdf(x, shape, scale=scale)

plt.figure(figsize=(8, 7))
plt.plot(x, pdf, color='darkblue', linestyle='-', linewidth=3)
plt.fill_between(x_fill, lognorm.pdf(x_fill, shape, scale=scale), color='red', alpha=0.7)
plt.xlabel('x', fontsize=16, labelpad=15)
plt.ylabel('p(x)', fontsize=16, labelpad=15)
plt.xlim(0, 5)
plt.ylim(0, 1.2 * max(pdf))
plt.legend([r'probability density function', r'$P(X < 0.5)$'], loc='upper right', fontsize=13, framealpha=0.8)
plt.tight_layout()
plt.savefig('pdf.png', dpi=600)
plt.show()


