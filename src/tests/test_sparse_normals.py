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


def sample_sparse_normals(r, n, be, sig):
    mu = np.sqrt(2 * r * np.log(n))
    ep = n ** -be
    idcs1 = np.random.rand(n) < ep / 2
    idcs2 = np.random.rand(n) < ep / 2

    Z1 = np.random.randn(n)
    Z2 = np.random.randn(n)

    Z1[idcs1] = sig*Z1[idcs1] + mu
    Z2[idcs2] = sig*Z2[idcs2] + mu
    Z = (Z1 - Z2)/np.sqrt(2)
    return Z

def test_sparse_normals(r, n, be, sig):
    Z = sample_sparse_normals(r, n, be, sig)
    pvals = 2*norm.cdf(- np.abs(Z))
    mt = MultiTest(pvals)
    fish = mt.fisher()
    return {'hc' : mt.hc(GAMMA)[0],
            'hcstar' : mt.hc_star(GAMMA)[0],
            'berkjones' : mt.berkjones(gamma=GAMMA),
            'berkjones_plus': mt.berkjones_plus(gamma=GAMMA),
            'bonf': mt.minp(),
            'fdr': mt.fdr()[0],
            'fisher': fish[0] / len(pvals)
            }
    
def test_feature_selection(r, n, be, sig):
    Z = sample_sparse_normals(r, n, be, sig)
    pvals = 2*norm.cdf(- np.abs(Z))
    mt = MultiTest(pvals)

    alpha = 0.05
    delta_HC = pvals <= mt.hc(GAMMA)[1] 
    delta_BJ = pvals <= mt.berkjones_threshold(gamma=GAMMA)
    delta_BH = pvals <= mt.fdr_threshold(fdr_param=alpha)
    delta_bonf = pvals <= alpha / n
    
    return {'delta_HC' : delta_HC,
            'delta_BJ' : delta_BJ,
            'delta_BH' : delta_BH,
            'delta_bonf' : delta_bonf
            }


def many_tests(n, be, r, sig, nMonte):
    # Test :nMonte: times

    lo_res = {}
    lo_num_features = {}
    for key in ['hc', 'hcstar', 'berkjones', 'berkjones_plus', 'bonf', 'fdr', 'fisher']:
        lo_res[key] = []
    for key in ['delta_HC', 'delta_BJ', 'delta_BH', 'delta_bonf']:
        lo_num_features[key] = []

    print(f"\n\nTesting with parameters: r={r}, n={n}, be={be}, sig={sig}")
    for _ in tqdm(range(nMonte)):
        res = test_sparse_normals(r, n, be, sig)
        for key in res:
            lo_res[key]+=[res[key]]
        res_features = test_feature_selection(r, n, be, sig)
        for key in res_features:
            lo_num_features[key] += [np.sum(res_features[key])]
    return lo_res, lo_num_features

lo_res, lo_num_features = many_tests(n=1000, be=0.75, r=0, sig=1, nMonte=10000)

print("Avg(HC) = ", np.mean(lo_res['hc']))
print("Avg(HCstar) = ", np.mean(lo_res['hcstar']))
print("Avg(BerkJones) = ", np.mean(lo_res['berkjones']))
print("Avg(BerkJonesPlus) = ", np.mean(lo_res['berkjones_plus']))
print("Avg(Bonf) = ", np.mean(lo_res['bonf']))
print("Avg(FDR) = ", np.mean(lo_res['fdr']))
print("Avg(Fisher) = ", np.mean(lo_res['fisher']))

print("# features HC = ", np.mean(lo_num_features['delta_HC']))
print("# features BerkJones = ", np.mean(lo_num_features['delta_BJ']))
print("# features BenjaminiHochberg = ", np.mean(lo_num_features['delta_BH']))
print("# features Bonferroni = ", np.mean(lo_num_features['delta_bonf']))


assert(np.abs(np.mean(lo_res['hc']) - 1.33) < .15)
assert(np.abs(np.mean(lo_res['hcstar']) - 1.29) < .15)
assert(np.abs(np.mean(lo_res['berkjones']) - 3.9) < .15)
assert(np.abs(np.mean(lo_res['bonf']) - 7.5) < 1)
assert(np.abs(np.mean(lo_res['fdr']) - 1) < .15)
assert(np.abs(np.mean(lo_res['fisher']) - 2) < .15)

lo_res, lo_num_features = many_tests(n=1000, be=0.75, r=1, sig=1, nMonte=10000)

print("Avg(HC) = ", np.mean(lo_res['hc']))
print("Avg(HCstar) = ", np.mean(lo_res['hcstar']))
print("Avg(BerkJones) = ", np.mean(lo_res['berkjones']))
print("Avg(BerkJonesPlus) = ", np.mean(lo_res['berkjones_plus']))
print("Avg(Bonf) = ", np.mean(lo_res['bonf']))
print("Avg(FDR) = ", np.mean(lo_res['fdr']))
print("Avg(Fisher) = ", np.mean(lo_res['fisher']))

print("# features HC = ", np.mean(lo_num_features['delta_HC']))
print("# features BerkJones = ", np.mean(lo_num_features['delta_BJ']))
print("# features BenjaminiHochberg = ", np.mean(lo_num_features['delta_BH']))
print("# features Bonferroni = ", np.mean(lo_num_features['delta_bonf']))


assert(np.abs(np.mean(lo_res['hc']) - 1.72) < .15)
assert(np.abs(np.mean(lo_res['hcstar']) - 1.69) < .15)
assert(np.abs(np.mean(lo_res['berkjones']) - 4.77) < 1)
assert(np.abs(np.mean(lo_res['bonf']) - 8.775) < 1)
assert(np.abs(np.mean(lo_res['fdr']) - 2.9) < .2)
assert(np.abs(np.mean(lo_res['fisher']) - 2) < .15)

lo_res, lo_num_features = many_tests(n=1000, be=0.75, r=0.9, sig=1, nMonte=10000)

print("Avg(HC) = ", np.mean(lo_res['hc']))
print("Avg(HCstar) = ", np.mean(lo_res['hcstar']))
print("Avg(BerkJones) = ", np.mean(lo_res['berkjones']))
print("Avg(BerkJonesPlus) = ", np.mean(lo_res['berkjones_plus']))
print("Avg(Bonf) = ", np.mean(lo_res['bonf']))
print("Avg(FDR) = ", np.mean(lo_res['fdr']))
print("Avg(Fisher) = ", np.mean(lo_res['fisher']))

print("# features HC = ", np.mean(lo_num_features['delta_HC']))
print("# features BerkJones = ", np.mean(lo_num_features['delta_BJ']))
print("# features BenjaminiHochberg = ", np.mean(lo_num_features['delta_BH']))
print("# features Bonferroni = ", np.mean(lo_num_features['delta_bonf']))


assert(np.abs(np.mean(lo_res['hc']) - 1.9) < .15)
assert(np.abs(np.mean(lo_res['hcstar']) - 1.8) < .15)
assert(np.abs(np.mean(lo_res['berkjones']) - 5) < 1)
assert(np.abs(np.mean(lo_res['bonf']) - 9.26) < 1)
assert(np.abs(np.mean(lo_res['fdr']) - 2.58) < .25)
assert(np.abs(np.mean(lo_res['fisher']) - 2) < .15)
