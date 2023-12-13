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

def test_sparse_normals(r, n, be, sig):
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

    _hc = MultiTest(pvals)

    fish = _hc.fisher()

    #import pdb; pdb.set_trace()

    return {'hc' : _hc.hc(GAMMA)[0],
            'hcstar' : _hc.hc_star(GAMMA)[0],
            'bj' : _hc.berk_jones(),
            'bonf': _hc.minp(),
            'fdr': _hc.fdr()[0],
            'fisher': fish[0] / len(pvals)
            }
    

def ManyTests(n, be, r, sig, nMonte):
    # Test :nMonte: times

    lo_res = {}
    for key in ['hc', 'hcstar', 'bj', 'bonf', 'fdr', 'fisher']:
        lo_res[key] = []

    print(f"\n\nTesting with parameters: r={r}, n={n}, be={be}, sig={sig}")
    for itr in tqdm(range(nMonte)):
        res = test_sparse_normals(r, n, be, sig)
        for key in res:
            lo_res[key]+=[res[key]]
    return lo_res

lo_res = ManyTests(n=1000, be=0.75, r=0, sig=1, nMonte=10000)

print("Avg(HC) = ", np.mean(lo_res['hc']))
print("Avg(HCstar) = ", np.mean(lo_res['hcstar']))
print("Avg(BerkJones) = ", np.mean(lo_res['bj']))
print("Avg(Bonf) = ", np.mean(lo_res['bonf']))
print("Avg(FDR) = ", np.mean(lo_res['fdr']))
print("Avg(Fisher) = ", np.mean(lo_res['fisher']))

assert(np.abs(np.mean(lo_res['hc']) - 1.33) < .15)
assert(np.abs(np.mean(lo_res['hcstar']) - 1.29) < .15)
assert(np.abs(np.mean(lo_res['bj']) - 3.9) < .15)
assert(np.abs(np.mean(lo_res['bonf']) - 7.5) < 1)
assert(np.abs(np.mean(lo_res['fdr']) - 1) < .15)
assert(np.abs(np.mean(lo_res['fisher']) - 2) < .15)


lo_res = ManyTests(n=1000, be=0.75, r=1, sig=1, nMonte=10000)

print("Avg(HC) = ", np.mean(lo_res['hc']))
print("Avg(HCstar) = ", np.mean(lo_res['hcstar']))
print("Avg(BerkJones) = ", np.mean(lo_res['bj']))
print("Avg(Bonf) = ", np.mean(lo_res['bonf']))
print("Avg(FDR) = ", np.mean(lo_res['fdr']))
print("Avg(Fisher) = ", np.mean(lo_res['fisher']))

assert(np.abs(np.mean(lo_res['hc']) - 1.72) < .15)
assert(np.abs(np.mean(lo_res['hcstar']) - 1.69) < .15)
assert(np.abs(np.mean(lo_res['bj']) - 4.77) < 1)
assert(np.abs(np.mean(lo_res['bonf']) - 8.775) < 1)
assert(np.abs(np.mean(lo_res['fdr']) - 2.9) < .2)
assert(np.abs(np.mean(lo_res['fisher']) - 2) < .15)

lo_res = ManyTests(n=1000, be=0.75, r=0.9, sig=1, nMonte=10000)

print("Avg(HC) = ", np.mean(lo_res['hc']))
print("Avg(HCstar) = ", np.mean(lo_res['hcstar']))
print("Avg(BerkJones) = ", np.mean(lo_res['bj']))
print("Avg(Bonf) = ", np.mean(lo_res['bonf']))
print("Avg(FDR) = ", np.mean(lo_res['fdr']))
print("Avg(Fisher) = ", np.mean(lo_res['fisher']))

assert(np.abs(np.mean(lo_res['hc']) - 1.9) < .15)
assert(np.abs(np.mean(lo_res['hcstar']) - 1.8) < .15)
assert(np.abs(np.mean(lo_res['bj']) - 5) < 1)
assert(np.abs(np.mean(lo_res['bonf']) - 9.26) < 1)
assert(np.abs(np.mean(lo_res['fdr']) - 2.58) < .25)
assert(np.abs(np.mean(lo_res['fisher']) - 2) < .15)
