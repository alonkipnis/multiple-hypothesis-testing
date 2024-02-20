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
ALPHA = 0.05

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
    
def test_feature_selection(r, n, be, sig):
    Z = sample_sparse_normals(r, n, be, sig)
    pvals = 2*norm.cdf(- np.abs(Z))
    mt = MultiTest(pvals)

    delta_HC = pvals <= mt.hc(GAMMA)[1] 
    delta_BJ = pvals <= mt.berkjones_threshold(gamma=GAMMA)
    delta_BH = pvals <= mt.fdr()[1]
    delta_FDR = pvals <= mt.fdr_control(fdr_param=2 * ALPHA)
    delta_bonf = pvals <= ALPHA / n
    
    return {'delta_HC' : delta_HC,
            'delta_BJ' : delta_BJ,
            'delta_BH' : delta_BH,
            'delta_FDR' : delta_FDR,
            'delta_bonf' : delta_bonf
            }

def many_tests(n, be, r, sig, nMonte):
    # Test :nMonte: times

    lo_num_features = {}
    for key in ['delta_HC', 'delta_BJ', 'delta_BH', 'delta_FDR', 'delta_bonf']:
        lo_num_features[key] = []

    print(f"\n\nTesting with parameters: r={r}, n={n}, be={be}, sig={sig}")
    for _ in tqdm(range(nMonte)):
        res_features = test_feature_selection(r, n, be, sig)
        for key in res_features:
            lo_num_features[key] += [np.mean(res_features[key])]
    return lo_num_features

n = 100
beta = 0.7
r = 2
nMonte = 10000

lo_num_features = many_tests(n=n, be=beta, r=r, sig=1, nMonte=nMonte)

print("# Proportion of truely affected featres = ", n**(-beta))
print("# features HC = ", np.mean(lo_num_features['delta_HC']))
print("# features BerkJones = ", np.mean(lo_num_features['delta_BJ']))
print("# features BenjaminiHochberg = ", np.mean(lo_num_features['delta_BH']))
print("# features Bonferroni = ", np.mean(lo_num_features['delta_bonf']))
print(f"# features FDR (alpha={ALPHA}) = ", np.mean(lo_num_features['delta_FDR']))

