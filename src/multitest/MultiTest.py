import numpy as np
from scipy.stats import beta, chi2
from matplotlib import pyplot as plt

MIN_PVAL = 1e-20


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
        hc       HC and P-value attaining it
        hc_star  more stable version of HC (HCdagger in [1])
        hc_jin   a version of HC from 
                [2] Jiashun Jin and Wanjie Wang, "Influential features PCA for
                 high dimensional clustering"

    Todo:
      Implement Feature selection procedures: HC-thresholding, FDR, BJ, Sims
      The idea is to return a mask based on the P-values
      Perhaps implement it in a different module dedicated to feature selection?

    """

    def __init__(self, pvals, stbl=True):

        self._N = len(pvals)
        assert (self._N > 0)

        self._EPS = 1 / (1e2 + self._N ** 2)
        self._istar = 1

        self._pvals = np.sort(np.asarray(pvals.copy()))
        self._uu = np.linspace(1 / self._N, 1, self._N)
        self._uu[-1] -= self._EPS # we assume that the largest P-value
                                  # has no effect on the results
        if stbl:
            denom = np.sqrt(self._uu * (1 - self._uu))
        else:
            denom = np.sqrt(self._pvals * (1 - self._pvals))

        self._zz = np.sqrt(self._N) * (self._uu - self._pvals) / denom

        self._imin_star = np.argmax(self._pvals > (1 - self._EPS) / self._N)
        self._imin_jin = np.argmax(self._pvals > np.log(self._N) / self._N)

        self._gamma = np.log(self._N) / np.sqrt(self._N)  # for 'auto' setting
                            # this gamma may be too small when N is large

    def _calculate_hc(self, imin, imax):
        if imin > imax:
            return np.nan
        if imin == imax:
            self._istar = imin
        else:
            self._istar = np.argmax(self._zz[imin:imax]) + imin
        zMaxStar = self._zz[self._istar]
        return zMaxStar, self._pvals[self._istar]

    def hc(self, gamma='auto'):
        """
        Higher Criticism test statistic

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
        return self._calculate_hc(imin, imax)

    def hc_jin(self, gamma='auto'):
        """sample-adjusted higher criticism score from [2]

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
        return self._calculate_hc(imin, imax)

    def berk_jones(self, gamma=.45):
        """
        Exact Berk-Jones statistic

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

        max_i = max(1, int(gamma * N))

        spv = self._pvals[:max_i]
        ii = np.arange(1, max_i + 1)

        bj = spv[0]
        if len(spv) >= 1:
            BJpv = beta.cdf(spv, ii, N - ii + 1)
            Mplus = np.min(BJpv)
            Mminus = np.min(1 - BJpv)
            bj = np.minimum(Mplus, Mminus)

        return -np.log(np.maximum(bj, MIN_PVAL))

    def hc_star(self, gamma='auto'):
        """sample-adjusted higher criticism score

        Args:
        -----
        'gamma' : lower fraction of P-values to consider

        Returns:
        -------
        :HC_score:
        :P-value attaining it:

        """
        if gamma == 'auto': 
            gamma = self._gamma
        imin = self._imin_star
        imax = np.maximum(imin + 1,
                          int(np.floor(gamma * self._N + 0.5)))
        return self._calculate_hc(imin, imax)

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

        yy = np.sort(self._pvals)[imin:imax]
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
        return -np.log(np.maximum(self._pvals[0], MIN_PVAL))

    def fdr(self):
        """
        Maximal False-discovery rate functional 

        Returns:
            :corrected critical P-value:
            :critical P-value:
        """

        vals = self._pvals / self._uu
        istar = np.argmin(vals)
        return -np.log(np.maximum(vals[istar], MIN_PVAL)), self._pvals[istar]

    def fisher(self):
        """
        combine P-values using Fisher's method:

        Fs = sum(-2 log(pvals))

        (here n is the number of P-values)

        When pvals are uniform Fs ~ chi^2 with 2*len(pvals) degrees of freedom

        Returns:
            Fs:       Fisher's method statistics
            Fs_pval:  P-value of the assocaited chi-squared test
        """

        Fs = np.sum(-2 * np.log(np.maximum(self._pvals, MIN_PVAL)))
        chi2_pval = chi2.sf(Fs, df=2 * len(self._pvals))
        return Fs, chi2_pval
