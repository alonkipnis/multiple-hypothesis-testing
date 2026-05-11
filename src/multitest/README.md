# MultiTest — Global Tests for Multiple Hypothesis Testing

MultiTest provides several methods for combining p-values with a focus on
detecting rare and weak effects.

## Higher Criticism variants

Each HC variant is its own method; the standardization is encoded in the
method name rather than passed as a constructor argument.

| Method | Standardization | P-value range |
|---|---|---|
| `hc`        | Donoho-Jin 2008 [2] – theoretical uniform std (default) | (0, γ] |
| `hc_dj2004` | Donoho-Jin 2004 [1] – observed p-value std | (0, γ] |
| `hc_dj2008` | Donoho-Jin 2008 [2] – theoretical uniform std | (0, γ] |
| `hc_beta`   | Beta-distribution std | (0, γ] |
| `hc_star`   | Beta-distribution std | (1/n, γ] (HCdagger [1]) |

All HC methods share the same signature:

```python
hc_*(gamma='auto', return_threshold=False)
```

- `gamma` – upper fraction of sorted p-values to consider. Only p-values
  ranked in positions 1 through ⌊γ·n⌋ enter the statistic.
  When `gamma='auto'` the value `log(n)/sqrt(n)` is used: this focuses HC
  on the sparse-signal regime where it has the most power, since signals
  denser than ~1/√n are better handled by simpler tests.
- `return_threshold` – set to `True` to obtain `(hc_score, threshold_pval)`.

## Other methods

- `berkjones` / `berkjones_plus` – Berk-Jones statistic [3]
- `fdr` – False-discovery rate functional
- `fdr_control` – Benjamini-Hochberg FDR control
- `bonferroni` / `neg_log_minp` – Bonferroni-style inference
- `fisher` – Fisher's method

## Example

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

# HC score + p-value threshold
hc, hct = mt.hc(return_threshold=True)

ii = np.arange(n)
print(f"HC = {hc:.3f}")
print(f"Features below HC threshold: {ii[pvals <= hct]}")
```

## References

[1] Donoho, David L. and Jin, Jiashun. "Higher criticism for detecting sparse heterogeneous mixtures." *The Annals of Statistics* 32, no. 3 (2004): 962-994.  
[2] Donoho, David L. and Jin, Jiashun. "Higher criticism thresholding: Optimal feature selection when useful features are rare and weak." *PNAS*, 2008.  
[3] Moscovich, Amit, Boaz Nadler, and Clifford Spiegelman. "On the exact Berk-Jones statistics and their p-value calculation." *Electronic Journal of Statistics* 10 (2016): 2329-2354.  
[4] Donoho, David L. and Alon Kipnis. "Higher criticism to compare two large frequency tables, with sensitivity to possible rare and weak differences." *The Annals of Statistics* 50, no. 3 (2022): 1447-1472.  
[5] Kipnis, Alon. "Unification of rare/weak detection models using moderate deviations analysis and log-chisquared p-values." *Statistica Sinica*, 2025.