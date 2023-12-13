# MultiTest -- Global Tests for Multiple Hypothesis

MultiTest includes several techniques for multiple hypothesis testing:
- ``MultiTest.hc`` Higher Criticism
- ``MultiTest.hcstar`` Higher Criticism with limited range 
- ``MultiTest.hc_jin`` Higher Criticism with limited range proposed by Jiashun Jin
- ``MultiTest.berk_jones`` Berk-Jones test (actually -log(bj)) 
- ``MultiTest.fdr`` False-discovery rate with optimized rate parameter
- ``MultiTest.minp`` Minimal P-values (Bonferroni style inference) (actually -log(minp))
- ``MultiTest.fisher`` Fisher's method to combine P-values
 
All tests rejects for large values of the test statistics. 

## Example:
```
import numpy as np
from scipy.stats import norm
from multitest import MultiTest

p = 100
z = np.random.randn(p)
pvals = 2*norm.cdf(-np.abs(z)/2)

mtest = MultiTest(pvals)

hc, p_hct = mtest.hc(gamma = 0.3)
bj = mtest.berk_jones()

ii = np.arange(len(pvals))
print(f"HC = {hc}, Indices of P-values below HCT: {ii[pvals <= p_hct]}")
print(f"Berk-Jones = {bj}")
```
