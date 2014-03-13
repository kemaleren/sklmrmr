import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_arrays
from sklmrmr._mrmr import _mrmr, MAXREL, MID, MIQ

from sklmrmr.bayesian_blocks import bayesian_blocks

__all__ = ['MRMR']


class MRMR(BaseEstimator, SelectorMixin):
    """Maximum Relevance Minimum Redundancy (MRMR) feature selection.

    Selects features with high mutual information with labels ``y``
    and low mutual information among themselves.

    MRMR can only handle discrete features and labels, because it
    needs to compute mutual information. It can optionally try to
    discretize continuous features.

    Parameters
    ----------
    k : int
        Number of features to select. Defaults to half of the given features.

    method : string
        One of 'mid', 'maxrel', or 'miq'.

    normalize : bool
        Whether to use normalized mutual information.

    discretize : bool
        Whether to discretize ``y`` and columns of ``X`` if necessary.
        If ``discretize`` is False and ``X`` or ``y`` cannot safely be
        converted to integers, throws an exception.

    Attributes
    ----------
    n_features_ : int
        Number of features selected. May be less than ``k``.

    support_ : ndarray
        ``support[i]`` is True if feature ``i`` was selected.

    ranking_ : ndarray
        ``ranking[i]`` is order in which feature ``i`` was selected,
        starting from 1.

    selected_ = ndarray
        Indices of features that were selected, in order.

    """
    methods = {'maxrel': MAXREL, 'mid': MID, 'miq': MIQ}
    warn_limit = 1000

    def __init__(self, k=None, method='mid', normalize=False,
                 discretize=False):
        self.k = k
        self.method = method.lower()
        self.normalize = normalize
        self.discretize = discretize

    def _validate(self):
        if self.method not in self.methods:
            raise ValueError("Unknown method: {}. Method must be one"
                             " of {}".format(self.method, self.methods.keys()))

    def _discretize(self, vector):
        vec_new = vector.astype(np.long)
        if np.all(vector == vec_new):
            return vec_new
        if len(set(vector)) == 1:
            return np.zeros(vector.shape, dtype=np.long)
        bin_edges = bayesian_blocks(vector)
        return np.digitize(vector, bin_edges)

    def fit(self, X, y):
        self._validate()
        X, y = check_arrays(X, y, sparse_format="csc")
        n_samples, n_features = X.shape

        # discretize X and y if necessary
        if np.issubdtype(X.dtype, float):
            X_new = X.astype(np.long)
            if np.any(X_new != X):
                if not self.discretize:
                    raise ValueError('MRMR does not support continuous values.'
                                     ' X could not safely be converted to'
                                     ' integers, and ``discretize`` is False')
                X = np.apply_along_axis(self._discretize, axis=0, arr=X)

        if np.issubdtype(y.dtype, float):
            y_new = y.astype(np.long)
            if np.any(y_new != y):
                if not self.discretize:
                    raise ValueError('MRMR does not support continuous values.'
                                     ' y could not safely be converted to'
                                     ' integers, and ``discretize`` is False')
                y = self._discretize(y)

        if self.k is None:
            k = n_features // 2
        else:
            k = self.k

        X_classes = np.array(list(set(X.reshape((n_samples * n_features,)))))
        y_classes = np.array(list(set(y.reshape((n_samples,)))))

        if len(X_classes) > self.warn_limit:
            print('Warning: X contains {} discrete values. MRMR may'
                  ' run slow'.format(len(X_classes)))
        if len(y_classes) > self.warn_limit:
            print('Warning: y contains {} discrete values. MRMR may'
                  ' run slow'.format(len(y_classes)))

        method = self.methods[self.method]
        idxs, _ = _mrmr(n_samples, n_features, y.astype(np.long),
                        X.astype(np.long), y_classes.astype(np.long),
                        X_classes.astype(np.long), y_classes.shape[0],
                        X_classes.shape[0], k, method, self.normalize)

        support_ = np.zeros(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int) + k

        support_[idxs] = True
        for i, idx in enumerate(idxs, start=1):
            ranking_[idx] = i

        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_
        self.selected_ = np.argsort(self.ranking_)[:self.n_features_]

        return self

    def _get_support_mask(self):
        return self.support_
