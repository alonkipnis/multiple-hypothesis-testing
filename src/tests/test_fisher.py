import sys
sys.path.append("../")
from src.multitest import MultiTest
import numpy as np
from tqdm import tqdm
from scipy.stats import norm

"""
Here we create two multivariate normals with rare
and weak differences in their means. 
"""

GAMMA = 'auto'

n = 1000
mu = 1

Z = np.random.randn(n)
print("Under null:")
pvals = 2*norm.cdf(- np.abs(Z))
fisher_stat, fisher_pvalue = MultiTest(pvals).fisher()
print("\tFisher stat:", fisher_stat)
print("\tFisher pvalue:", fisher_pvalue)

assert np.abs(fisher_stat - 2 * n ) < 4*np.sqrt(n)

print("Under alternative:")
X = Z + mu
pvals = 2*norm.cdf(- np.abs(X))
fisher_stat, fisher_pvalue = MultiTest(pvals).fisher()
print("\tFisher stat:", fisher_stat)
print("\tFisher pvalue:", fisher_pvalue)

print(2 * n + n * mu**2 + 0.5*n*np.log(1+mu))
print(5 * np.sqrt(n + 4 * n * mu**2))

assert np.abs(fisher_stat - 2 * n - n * mu**2 - 0.5*n*np.log(1+mu) ) <  5 * np.sqrt(n + 4 * n * mu**2)