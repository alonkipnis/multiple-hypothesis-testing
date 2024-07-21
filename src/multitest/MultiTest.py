import numpy as np
from scipy.stats import beta, chi2
from matplotlib import pyplot as plt


class MultiTest(object):
    """
    Mulitple testing class for P-values with a focus on tests for rare and weak effects.
    The package implemnents the following tests:
    - Higher criticism [1] and varaints from [2] and [3], including HC threshold from [2]
    - Berk-Jones [4]
    - Fisher's method
    - Bonferroni type inference
    - Family-wise significant testing using FDR control
    - P-value selection using Benjamini-Hochberg's FDR control

    References:
    [1] Donoho, D. L. and Jin, J.,
     "Higher criticism for detecting sparse hetrogenous mixtures", 
     Annals of Stat. 2004
    [2] Donoho, D. L. and Jin, J. "Higher critcism thresholding: Optimal 
    feature selection when useful features are rare and weak", proceedings
    of the national academy of sciences, 2008.
    [3] Jiashun Jin and Wanjie Wang, "Influential features PCA for
                 high dimensional clustering"
    [4] Amit Moscovich, Boaz Nadler, and Clifford Spiegelman. "On the exact Berk-Jones statistics
      and their p-value calculation." Electronic Journal of Statistics. 10 (2016): 2329-2354.

    ========================================================================

    Args:
    -----
        pvals    list of p-values. P-values that are np.nan are exluded.
        stbl     normalize by expected P-values (stbl=True) or observed
                 P-values (stbl=False). stbl=True was suggested in [2] while 
                 [1] studied the case stbl=False. 
        
    Methods :
    -------
        hc       HC and P-value attaining it 
        hc_star  Only consider P-values larger than 1/n. (HCdagger in [1]) 
        hc_jin   Only consider P-values larger than log(n)/n. This variant was introduced in [3]
        berkjones  Exact Berk-Jones statistic (see [4])
    """

    def __init__(self, pvals, stbl=True):

        self._N = len(pvals)
        assert (self._N > 0)

        self._EPS = 1 / (1e4 + self._N ** 2)
        self._istar = 0

        self._sorted_pvals = np.sort(np.asarray(pvals.copy()))  # sorted P-values
        self._uu = np.linspace(1 / self._N, 1, self._N)
        self._uu[-1] -= self._EPS # we assume that the largest P-value
                                  # does no effect the results
        if stbl:
            denom = np.sqrt(self._uu * (1 - self._uu))
        else:
            denom = np.sqrt(self._sorted_pvals * (1 - self._sorted_pvals))

        self._zz = np.sqrt(self._N) * (self._uu - self._sorted_pvals) / denom

        self._imin_star = np.argmax(self._sorted_pvals > (1 - self._EPS) / self._N)
        self._imin_jin = np.argmax(self._sorted_pvals > np.log(self._N) / self._N)

        self._gamma = np.log(self._N) / np.sqrt(self._N)  # for 'auto' setting
                            # this gamma was suggested in [3]
                            # The rationalle is that HC is not useful for signal denser than 1/sqrt(n)

    def _evaluate_hc(self, imin, imax):
        if imin > imax:
            return np.nan
        if imin == imax:
            self._istar = imin
        else:
            self._istar = np.argmax(self._zz[imin:imax]) + imin
        zMaxStar = self._zz[self._istar]
        return zMaxStar, self._sorted_pvals[self._istar]

    def hc(self, gamma='auto'):
        """
        Higher Criticism test of [1]

        Args:
        -----
        gamma   lower fraction of P-values to consider

        Return:
        -------
        HC test score, P-value attaining it

        """
        imin = 0
        if gamma == 'auto': 
            gamma = self._gamma
        imax = np.maximum(imin, int(gamma * self._N + 0.5))
        return self._evaluate_hc(imin, imax)

    def hc_jin(self, gamma='auto'):
        """
        but only consider P-values larger than log(n)/n. 
        This variant was introduced in [3]/ 

        Args:
        -----
        gamma   lower fraction of P-values to consider

        Return:
        -------
        HC score, P-value attaining it
        """

        if gamma == 'auto': 
            gamma = self._gamma
        imin = self._imin_jin
        imax = np.maximum(imin + 1, int(np.floor(gamma * self._N + 0.5)))
        return self._evaluate_hc(imin, imax)

    def berkjones(self, gamma='auto', min_only=False):
        """
        Exact Berk-Jones statistic

        According to Moscovich, Nadler, Spiegelman. (2013). 
        On the exact Berk-Jones statistics and their p-value calculation

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
        Exact Berk-Jones statistic only lower-than-uniform P-values

        According to Moscovich, Nadler, Spiegelman. (2013). 
        On the exact Berk-Jones statistics and their p-value calculation

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

    def hc_star(self, gamma='auto'):
        """sample-adjusted higher criticism score

        Args:
        -----
        'gamma' : lower fraction of P-values to consider

        Returns:
        -------
        HC score
        P-value attaining it

        """
        if gamma == 'auto': 
            gamma = self._gamma
        imin = self._imin_star
        imax = np.maximum(imin + 1, int(np.floor(gamma * self._N + 0.5)))
        return self._evaluate_hc(imin, imax)

    def hc_dashboard(self, gamma='auto'):
        """
        Illustrates HC over z-scores and sorted P-values.

        Args:
            gamma:  HC parameter

        Returns:
            fig: an illustration of HC value

        """
        if gamma == 'auto': 
            gamma = self._gamma

        hc, hct = self.hc(gamma)
        imin = 0
        N = self._N
        istar = self._istar

        imax = np.maximum(imin, int(gamma * N + 0.5))

        yy = np.sort(self._sorted_pvals)[imin:imax]
        zz = self._zz[imin:imax]
        xx = self._uu[imin:imax]

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
        return {'pvals': self._sorted_pvals,
                'u': self._uu,
                'z': self._zz,
                'imin_star': self._imin_star,
                'imin_jin': self._imin_jin,
                }

    def bonfferoni(self):
        """
        Bonferroni type inference
        """
        return self._sorted_pvals[0] * self._N

    def minp(self):
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

        vals = self._sorted_pvals / self._uu
        istar = np.argmin(vals)
        return -np.log(vals[istar]), self._sorted_pvals[istar]

    def fdr_control(self, fdr_param=0.1):
        """
        Binjimini-Hochberg FDR control

        Args:
            fdr_param: False discovery rate parameter

        Returns:
            P-value p(i^*) such that the the proportion of false discoveries in {p(i) <= p(i^*)} is 
            samller in expectation than fdr_param
        """

        vals = self._sorted_pvals / self._uu
        indicator = vals > fdr_param # example 0 0 0 1 1 1 1 1 1
        istar = np.argmax(indicator)-1           # first 1
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