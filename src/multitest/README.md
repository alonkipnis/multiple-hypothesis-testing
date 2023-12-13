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

p = 1000
eps = .01
mu = 2
I = np.random.rand(p) < eps
z = np.random.randn(p) * (1 - I) + I * (np.random.randn(p) + mu)
pvals = 2*norm.cdf(-np.abs(z))

mtest = MultiTest(pvals)

f = mtest.fisher()
hc = mtest.hc()


print(f"Fisher's comination statistic = {f[0]}, p-value = {f[1]}")
print(f"Higher criticism statistic= {hc[0]}, Higher criticism threshold= {hc[1]}")
```
