import numpy as np
from scipy.stats import beta


class MultiTest(object):
    """
    Higher Criticism test 

    References:
    [1] Donoho, D. L. and Jin, J.,
     "Higher criticism for detecting sparse hetrogenous mixtures", 
     Annals of Stat. 2004
    [2] Donoho, D. L. and Jin, J. "Higher critcism thresholding: Optimal 
    feature selection when useful features are rare and weak", proceedings
    of the national academy of sciences, 2008.
    ========================================================================

    Args:
    -----
        pvals    list of p-values. P-values that are np.nan are exluded.
        stbl     normalize by expected P-values (stbl=True) or observed 
                 P-values (stbl=False). stbl=True was suggested in [2].
                 stbl=False in [1]. 
        gamma    lower fruction of p-values to use.
        
    Methods :
    -------
        HC       HC and P-value attaining it
        HCstar   sample adjustet HC (HCdagger in [1])
        HCjin    a version of HC from 
                [2] Jiashun Jin and Wanjie Wang, "Influential features PCA for
                 high dimensional clustering"


    Todo:
      implement Feature selection procedures: HC-thresholding, FDR, BJ, Sims
    """

    def __init__(self, pvals, stbl=True):

        self._N = len(pvals)
        assert (self._N > 0)

        EPS = 1 / self._N
        self._EPS = EPS
        self._istar = 1

        self._pvals = np.sort(np.asarray(pvals.copy()))
        self._uu = np.linspace(1 / self._N, 1 - EPS, self._N)

        if stbl:
            denom = np.sqrt(self._uu * (1 - self._uu))
        else:
            denom = np.sqrt(self._pvals * (1 - self._pvals))

        denom = np.maximum(denom, EPS)
        self._zz = np.sqrt(self._N) * (self._uu - self._pvals) / denom

        self._imin_star = np.argmax(self._pvals > (1 - self._EPS) / self._N)
        self._imin_jin = np.argmax(self._pvals > np.log(self._N) / self._N)

    def _calculate_hc(self, imin, imax):
        if imin > imax:
            return np.nan
        if imin == imax:
            self._istar = imin
        else:
            self._istar = np.argmax(self._zz[imin:imax]) + imin
        zMaxStar = self._zz[self._istar]
        return zMaxStar, self._pvals[self._istar]

    def hc(self, gamma=0.2):
        """
        Higher Criticism test statistic

        Args:
        -----
        'gamma' : lower fraction of P-values to consider

        Return:
        -------
        HC test score, P-value attaining it

        """
        imin = 0
        imax = np.maximum(imin, int(gamma * self._N + 0.5))
        return self._calculate_hc(imin, imax)

    def hc_jin(self, gamma=0.2):
        """sample-adjusted higher criticism score from [2]

        Args:
        -----
        'gamma' : lower fraction of P-values to consider

        Return:
        -------
        HC score, P-value attaining it

        """

        imin = self._imin_jin
        imax = np.maximum(imin + 1,
                          int(np.floor(gamma * self._N + 0.5)))
        return self._calculate_hc(imin, imax)

    def berk_jones(self, gamma=1):
        """
        Exact Berk-Jones statistic

        According to Moscovich, Nadler, Spiegelman. (2013). 
        On the exact Berk-Jones statistics and their p-value calculation

        Args:
        -----
        'gamma' : lower fraction of P-values to consider

        Return:
        -------
        -log(BJ) score (large values are significant) 
        (has a scaled chisquared distribution under the null)

        """

        N = self._N

        if N == 0:
            return np.nan, np.nan

        max_i = max(1, int(gamma * N))

        spv = self._pvals[:max_i]
        ii = np.arange(1, max_i + 1)

        bj = spv[0]
        p_th = spv[0]

        if len(spv) >= 1:
            BJpv = beta.cdf(spv, ii, N - ii + 1)
            Mplus = np.min(BJpv)
            Mminus = np.min(1 - BJpv)
            bj = np.minimum(Mplus, Mminus)

        return -np.log(bj + self._EPS / 1000)

    def hc_star(self, gamma=0.2):
        """sample-adjusted higher criticism score

        Args:
        -----
        'gamma' : lower fraction of P-values to consider

        Returns:
        -------
        :HC_score:
        :P-value attaining it:

        """

        imin = self._imin_star
        imax = np.maximum(imin + 1,
                          int(np.floor(gamma * self._N + 0.5)))
        return self._calculate_hc(imin, imax)

    def get_state(self):
        return {'pvals': self._pvals,
                'u': self._uu,
                'z': self._zz,
                'imin_star': self._imin_star,
                'imin_jin': self._imin_jin,
                }

    def minp(self):
        """
        Bonferroni type inference

        -log(minimal P-value)
        """
        return -np.log(self._pvals[0])

    def fdr(self):
        """
        False-discovery rate

        Returns:
            :corrected critical P-value:
            :critical P-value:
        """

        vals = self._pvals / self._uu
        self._istar = np.argmin(vals)
        return vals[self._istar], self._pvals[self._istar]

    def fisher(self):
        """
        combine P-values using Fisher's method:

        Fs = sum(-2 log(pvals))

        When pvals are uniform Fs ~ chi^2 with len(pvals) degrees of freedom

        Returns:
            :Fs:       Fisher's method statistics
            :degrees of freedom:   number of degrees of freedom
        """
        Fs = np.sum(-2 * np.log(self._pvals))
        return Fs, self._N
