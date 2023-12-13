import sys
sys.path.append("../")
from src.multitest import MultiTest
import numpy as np
from tqdm import tqdm
from scipy.stats import norm

from matplotlib import pyplot as plt

r = 1.
n = 1000
be = 0.7
sig = 1

mu = np.sqrt(2 * r * np.log(n))
ep = n ** -be
idcs1 = np.random.rand(n) < ep / 2
idcs2 = np.random.rand(n) < ep / 2

Z1 = np.random.randn(n)
Z2 = np.random.randn(n)

Z1[idcs1] = sig*Z1[idcs1] + mu
Z2[idcs2] = sig*Z2[idcs2] + mu

Z = (Z1 - Z2)/np.sqrt(2)
pvals = 2*norm.cdf(- np.abs(Z))

mt = MultiTest(pvals)
hc, hct = mt.hc()

mt.hc_dashboard(gamma=.3)
