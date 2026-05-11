# MultiTest — Global Tests for Multiple Hypothesis Testing

MultiTest provides several methods for combining p-values with a focus on
detecting rare and weak effects.

## Higher Criticism variants

Each HC variant is exposed as its own method so that the standardization
choice is explicit rather than a constructor argument.

| Method | Standardization | P-value range considered |
|---|---|---|
| `MultiTest.hc`        | Donoho-Jin 2008 [2] – theoretical uniform std (default) | (0, γ] |
| `MultiTest.hc_dj2004` | Donoho-Jin 2004 [1] – observed p-value std | (0, γ] |
| `MultiTest.hc_dj2008` | Donoho-Jin 2008 [2] – theoretical uniform std | (0, γ] |
| `MultiTest.hc_beta`   | Beta-distribution std | (0, γ] |
| `MultiTest.hc_star`   | Beta-distribution std | (1/n, γ] (HCdagger [1]) |

Every HC method accepts:
- `gamma` – upper fraction of sorted p-values to consider. Only the
  p-values ranked in positions 1 through ⌊γ·n⌋ (i.e. the smallest γ
  fraction) enter the HC statistic. Defaults to `'auto'` (see below).
- `return_threshold` – if `True`, returns `(hc_score, threshold_pval)`;
  otherwise returns just the score (default `False`).

### Default `gamma` and why it matters

When `gamma='auto'` the upper limit is set to

```
γ = log(n) / sqrt(n)
```

This keeps HC focused on the regime where it has the most power. Signals
denser than roughly 1/√n features are detectable by simpler methods (e.g.
the average z-score), so extending HC beyond γ = log(n)/√n adds noise
without gaining power. The log(n) factor provides a small safety margin
above the 1/√n threshold.

## Other methods

- `MultiTest.berkjones` / `berkjones_plus` – Berk-Jones statistic [3]
- `MultiTest.fdr` – False-discovery rate functional
- `MultiTest.fdr_control` – Benjamini-Hochberg FDR control
- `MultiTest.bonferroni` / `neg_log_minp` – Bonferroni-style inference
- `MultiTest.fisher` – Fisher's method to combine p-values

In all cases, reject the null for large values of the test statistic.

## Quick start

```python
import numpy as np
from scipy.stats import norm
from multitest import MultiTest

n = 100
z = np.random.randn(n)
pvals = 2 * norm.cdf(-np.abs(z))

mt = MultiTest(pvals)

# Default HC score
hc = mt.hc()

# HC score + threshold p-value
hc, hct = mt.hc(return_threshold=True)

# Berk-Jones
bj = mt.berkjones()

ii = np.arange(n)
print(f"HC = {hc:.3f}, features below HCT: {ii[pvals <= hct]}")
print(f"Berk-Jones = {bj:.3f}")
```

## Choosing an HC variant

- **`hc`** is the recommended starting point. It is an alias for `hc_dj2008`
  with the default `gamma` setting.
- **`hc_dj2008`** uses the expected (theoretical) uniform mean and std for
  normalization, matching the formulation in [2].
- **`hc_beta`** is nearly identical to `hc_dj2008` for large n but uses the
  exact beta-distribution moments of order statistics.
- **`hc_dj2004`** uses the *observed* p-value standard deviation as denominator.
  This makes it more sensitive to extreme p-values but also increases variance
  under the null.
- **`hc_star`** ignores p-values below 1/n (sample-size adjusted, HCdagger [1]).

## Use cases

This package was used to obtain evaluations reported in [4] and [5].

## References

[1] Donoho, David L. and Jin, Jiashun. "Higher criticism for detecting sparse heterogeneous mixtures." *The Annals of Statistics* 32, no. 3 (2004): 962-994.  
[2] Donoho, David L. and Jin, Jiashun. "Higher criticism thresholding: Optimal feature selection when useful features are rare and weak." *Proceedings of the National Academy of Sciences*, 2008.  
[3] Moscovich, Amit, Boaz Nadler, and Clifford Spiegelman. "On the exact Berk-Jones statistics and their p-value calculation." *Electronic Journal of Statistics* 10 (2016): 2329-2354.  
[4] Donoho, David L. and Alon Kipnis. "Higher criticism to compare two large frequency tables, with sensitivity to possible rare and weak differences." *The Annals of Statistics* 50, no. 3 (2022): 1447-1472.  
[5] Kipnis, Alon. "Unification of rare/weak detection models using moderate deviations analysis and log-chisquared p-values." *Statistica Sinica*, 2025.