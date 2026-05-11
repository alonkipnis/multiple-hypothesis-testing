import numpy as np
from scipy.stats import beta, chi2
from matplotlib import pyplot as plt


class MultiTest(object):
    f"""
    Multiple testing class for P-values with a focus on tests for rare and weak effects.
    The package implements the following tests:
    - Higher criticism [1] and variants from [2] 
    - Modifying upper limit of HC to log(n)/sqrt(n)
    - Modifying the demoninator of HC to std of the beta distribution of ordered P-values
    - Berk-Jones [3]
    - Fisher's method
    - Bonferroni type inference
    - Family-wise significant testing using FDR control
    - P-value selection using Benjamini-Hochberg's FDR control

    References:
    [1] Donoho, D. L. and Jin, J.,
     "Higher criticism for detecting sparse heterogeneous mixtures",
     Annals of Stat. 2004
    [2] Donoho, D. L. and Jin, J. "Higher criticism thresholding: Optimal
feature selection when useful features are rare and weak", proceedings
    of the national academy of sciences, 2008.
    [3] Amit Moscovich, Boaz Nadler, and Clifford Spiegelman. "On the exact Berk-Jones statistics
      and their p-value calculation." Electronic Journal of Statistics. 10 (2016): 2329-2354.

    ========================================================================

    Args:
    -----
        pvals   list of p-values. P-values that are np.nan are excluded.

    Methods:
    -------
        hc          Default HC (alias for hc_dj2008)
        hc_dj2004   HC with Donoho-Jin 2004 standardization [1]
        hc_dj2008   HC with Donoho-Jin 2008 standardization [2]
        hc_star     HC variant only considering P-values > 1/n (HCdagger in [1])
        hc_beta     HC with beta-distribution standardization
        berkjones   Exact Berk-Jones statistic (see [3])
    """

    def __init__(self, pvals):

        self._N = len(pvals)
        assert (self._N > 0)

        self._EPS = 1 / (1e4 + self._N ** 2)
        self._istar = 0

        self._sorted_pvals = np.sort(np.asarray(pvals.copy()))

        self._imin_star = np.argmax(self._sorted_pvals > (1 - self._EPS) / self._N)

        self._gamma = np.log(self._N) / np.sqrt(self._N)  
        # The rationale is particularly useful for detecting effect much rarer thatn sqrt{n} of the number of features. This upper truncation was not published, but seems to provide good results. 

    def hc(self, gamma='auto', return_threshold=False):
        """
        Default Higher Criticism (alias for hc_dj2008).

        Args:
        -----
        gamma             upper fraction of P-values to consider (default: log(n)/sqrt(n))
        return_threshold  if True, also return the P-value attaining HC

        Return:
        -------
        HC score (and P-value attaining it if return_threshold=True)
        """
        return self.hc_dj2008(gamma=gamma, return_threshold=return_threshold)


    def _get_zscores(self, standardization):
        """Compute z-scores for the given standardization type.

        Returns (uu, zz) where uu are the expected uniform quantiles and zz are
        the HC z-scores.
        """
        N = self._N
        spv = self._sorted_pvals

        if standardization == 'donoho-jin2008':
            uu = np.linspace(1 / N, 1, N)
            uu[-1] -= self._EPS
            std = np.sqrt(uu * (1 - uu) / N)
        elif standardization == 'donoho-jin2004':
            uu = np.linspace(1 / N, 1, N)
            uu[-1] -= self._EPS
            std = np.sqrt(spv * (1 - spv) / N)
        else:  # 'beta-std'
            uu = np.linspace(1 / (N + 1), 1 - 1 / (N + 1), N)
            std = np.sqrt(uu * (1 - uu) / (N + 2))

        return uu, (uu - spv) / std

    def _evaluate_hc(self, zz, imin, imax, return_threshold=False):
        if imin > imax:
            return np.nan
        if imin == imax:
            self._istar = imin
        else:
            self._istar = np.argmax(zz[imin:imax]) + imin
        hc_score = zz[self._istar]
        if return_threshold:
            return hc_score, self._sorted_pvals[self._istar]
        return hc_score

    def hc_dj2004(self, gamma='auto', return_threshold=False):
        """
        Higher Criticism with Donoho-Jin 2004 standardization [1].

        Z-scores use the observed P-value std for normalization.

        Args:
        -----
        gamma             lower fraction of P-values to consider
        return_threshold  if True, also return the P-value attaining HC

        Return:
        -------
        HC score (and P-value attaining it if return_threshold=True)
        """
        _, zz = self._get_zscores('donoho-jin2004')
        imin = 0
        if gamma == 'auto':
            gamma = self._gamma
        imax = np.maximum(imin, int(gamma * self._N + 0.5))
        return self._evaluate_hc(zz, imin, imax, return_threshold)

    def hc_dj2008(self, gamma='auto', return_threshold=False):
        """
        Higher Criticism with Donoho-Jin 2008 standardization [2].

        Z-scores use the expected P-value mean and std from the uniform distribution.

        Args:
        -----
        gamma             lower fraction of P-values to consider
        return_threshold  if True, also return the P-value attaining HC

        Return:
        -------
        HC score (and P-value attaining it if return_threshold=True)
        """
        _, zz = self._get_zscores('donoho-jin2008')
        imin = 0
        if gamma == 'auto':
            gamma = self._gamma
        imax = np.maximum(imin, int(gamma * self._N + 0.5))
        return self._evaluate_hc(zz, imin, imax, return_threshold)

    def hc_beta(self, gamma='auto', return_threshold=False):
        """
        Higher Criticism with beta-distribution standardization.

        Z-scores use mean and std matched to the beta distribution of ordered P-values.

        Args:
        -----
        gamma             lower fraction of P-values to consider
        return_threshold  if True, also return the P-value attaining HC

        Return:
        -------
        HC score (and P-value attaining it if return_threshold=True)
        """
        _, zz = self._get_zscores('beta-std')
        imin = 0
        if gamma == 'auto':
            gamma = self._gamma
        imax = np.maximum(imin, int(gamma * self._N + 0.5))
        return self._evaluate_hc(zz, imin, imax, return_threshold)


    def berkjones(self, gamma='auto', min_only=False):
        """
        Exact Berk-Jones statistic. See [3].

        Args:
        -----
        gamma  lower fraction of P-values to consider. 

        Return:
        -------
        -log(BJ) score (large values are significant) 
        (has a scaled chisquared distribution under the null)

        """

        N = self._N
        if N == 0:
            return np.nan, np.nan
        
        if gamma == 'auto': 
            gamma = self._gamma

        max_i = max(1, int(gamma * N))

        spv = self._sorted_pvals[:max_i]
        ii = np.arange(1, max_i + 1)

        bj = spv[0]
        if len(spv) >= 1:
            BJpv = beta.cdf(spv, ii, N - ii + 1)
            Mplus = np.min(BJpv)
            Mminus = np.min(1 - BJpv)
            if min_only: #only use BJ+ for the score
                bj = Mplus 
            else:
                bj = np.minimum(Mplus, Mminus) 
        return -np.log(bj)
    
    def berkjones_plus(self, gamma='auto'):
        """
        Exact Berk-Jones statistic only lower-than-uniform P-values (See [3])

        Args:
        -----
        gamma  lower fraction of P-values to consider. Better to pick
               gamma < .5 or far below 1 to avoid p-values that are one

        Return:
        -------
        -log(BJ) score (large values are significant) 
        (has a scaled chisquared distribution under the null)

        """

        N = self._N

        if N == 0:
            return np.nan, np.nan
        
        if gamma == 'auto': 
            gamma = self._gamma

        max_i = max(1, int(gamma * N))

        spv = self._sorted_pvals[:max_i]
        ii = np.arange(1, max_i + 1)

        Mplus = spv[0]
        if len(spv) >= 1:
            BJpv = beta.cdf(spv, ii, N - ii + 1)
            Mplus = np.min(BJpv)
            
        return -np.log(Mplus)

    def berkjones_threshold(self, gamma='auto'):
        """
        Use the Berk-Jones statistic to find a threshold for P-values in 
        a manner analogous to the HC threshold of [2]

        Args:
        -----
        gamma  lower fraction of P-values to consider

        Return:
        -------
        P-value attaining the minimum in Mplus
        """

        N = self._N
        if N == 0:
            return np.nan, np.nan
        
        if gamma == 'auto': 
            gamma = self._gamma

        max_i = max(1, int(gamma * N))
        spv = self._sorted_pvals[:max_i]
        ii = np.arange(1, max_i + 1)

        istar = 0
        if len(spv) >= 1:
            BJpv = beta.cdf(spv, ii, N - ii + 1)
            istar = np.argmin(BJpv)
        return spv[istar]  # P-value attaining the minimum in Mplus

    def hc_star(self, gamma='auto', return_threshold=False):
        """
        Sample-adjusted Higher Criticism from [1] (HCdagger).

        Only considers P-values larger than 1/n, using beta-distribution
        standardization.

        Args:
        -----
        gamma             lower fraction of P-values to consider
        return_threshold  if True, also return the P-value attaining HC

        Returns:
        -------
        HC score (and P-value attaining it if return_threshold=True)
        """
        _, zz = self._get_zscores('beta-std')
        if gamma == 'auto':
            gamma = self._gamma
        imin = self._imin_star
        imax = np.maximum(imin + 1, int(np.floor(gamma * self._N + 0.5)))
        return self._evaluate_hc(zz, imin, imax, return_threshold)

    def hc_dashboard(self, gamma='auto'):
        """
        Illustrates HC over z-scores and sorted P-values.

        Uses beta-distribution standardization (hc_beta).

        Args:
            gamma:  HC parameter

        Returns:
            fig: an illustration of HC value

        """
        if gamma == 'auto':
            gamma = self._gamma

        hc, hct = self.hc_beta(gamma, return_threshold=True)
        imin = 0
        N = self._N
        istar = self._istar

        imax = np.maximum(imin, int(gamma * N + 0.5))

        uu, zz_full = self._get_zscores('beta-std')
        yy = np.sort(self._sorted_pvals)[imin:imax]
        zz = zz_full[imin:imax]
        xx = uu[imin:imax]

        ax = plt.subplot(211)

        ax.stem(xx, yy, markerfmt='.')
        ax.plot([(istar + 1) / N, (istar + 1) / N], [0, hct], '--r', alpha=.75)
        ax.set_ylabel('p-value', fontsize=14)
        ax.set_title('Sorted P-values')
        ax.set_xlim([0, imax / N])
        ax.set_xlabel('i/n', fontsize=16)

        labels = ax.get_xticklabels()
        labels[-1].set_text(r"$\gamma_0=$" + labels[-1]._text)
        ax.set_xticks(ticks=[l._x for l in labels], labels=labels)

        # second plot
        ax = plt.subplot(212)
        ax.plot(xx, zz)
        ymin = np.min(zz) * 1.1
        ax.plot([(istar + 1) / N, (istar + 1) / N], [ymin, hc], '--r', alpha=.75)

        ax.plot([ymin, (istar + 1) / N], [hc, hc], '--r', alpha=.75)
        ax.text(-0.01, hc, r'$HC$', horizontalalignment='center', fontsize=14,
                bbox=dict(boxstyle="round",
                          ec=(1., 1, 1),
                          fc=(1., 1, 1),
                          alpha=0.5,
                          ))

        ax.set_ylabel('z-score', fontsize=14)

        ax.grid(True)
        ax.set_xlim([0, imax / N])
        ax.set_xlabel('i/n', fontsize=16)

        label = ax.get_xticklabels()[-1]
        label.set_text(r"$\gamma_0=$" + label._text)
        ax.set_xticks(ticks=[label._x, (istar + 1) / N], labels=[label, str(np.round((istar + 1) / N, 2))])

        fig = plt.gcf()
        fig.set_size_inches(10, 10, forward=True)

        plt.show()
        return fig

    def get_state(self):
        uu, zz = self._get_zscores('beta-std')
        return {'pvals': self._sorted_pvals,
                'u': uu,
                'z': zz,
                'imin_star': self._imin_star,
                }

    def bonferroni(self):
        """
        Bonferroni type inference
        """
        return self._sorted_pvals[0] * self._N

    def neg_log_minp(self):
        """
        Bonferroni type inference

        Returns:
        -log(minimal P-value)
        """
        return -np.log(self._sorted_pvals[0])

    def fdr(self):
        """
        Maximal False-discovery rate functional 

        Returns:
            -log(p(i^*)), p(i^*) where i^* is the index of the
            critical P-value
        """
        uu, _ = self._get_zscores('beta-std')
        vals = self._sorted_pvals / uu
        istar = np.argmin(vals)
        return -np.log(vals[istar]), self._sorted_pvals[istar]

    def fdr_control(self, fdr_param=0.1):
        """
        Binjimini-Hochberg FDR control

        Args:
            fdr_param: False discovery rate parameter

        Returns:
            P-value p(i^*) such that the the proportion of false discoveries in {p(i) <= p(i^*)} is 
            smaller in expectation than fdr_param
        """
        uu, _ = self._get_zscores('beta-std')
        vals = self._sorted_pvals / uu
        indicator = vals > fdr_param
        istar = np.argmax(indicator) - 1
        if istar < 0:
            return np.nan
        return self._sorted_pvals[istar]

    def fisher(self):
        """
        combine P-values using Fisher's method:

        Fs = sum(-2 log(pvals))

        (here n is the number of P-values)

        When pvals are uniform Fs ~ chi^2 with 2*len(pvals) degrees of freedom

        Returns:
            fisher_comb_stat       Fisher's method statistics
            chi2_pval              P-value of the assocaited chi-squared test
        """

        fisher_comb_stat = np.sum(-2 * np.log(self._sorted_pvals))
        chi2_pval = chi2.sf(fisher_comb_stat, df=2 * len(self._sorted_pvals))
        return fisher_comb_stat, chi2_pval