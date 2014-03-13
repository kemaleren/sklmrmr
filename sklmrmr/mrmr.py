import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_arrays
from sklmrmr._mrmr import _mrmr, MAXREL, MID, MIQ

__all__ = ['MRMR']


class MRMR(BaseEstimator, SelectorMixin):
    """Maximum Relevance Minimum Redundancy (MRMR) feature selection.

    Selects features with high mutual information with labels ``y``
    and low mutual information among themselves.

    Parameters
    ----------
    k : int
        Number of features to select. Defaults to half of the given features.

    method : string
        One of 'mid', 'maxrel', or 'miq'.

    normalize : bool
        Whether to use normalized mutual information.

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

    def __init__(self, k=None, method='mid', normalize=False):
        self.method = method
        self.k = k
        self.method = method
        self.normalize = normalize

    def _validate(self):
        if self.method not in self.methods:
            raise ValueError("Unknown method: {}. Method must be one"
                             " of {}".format(self.method, self.methods.keys()))

    def fit(self, X, y):
        self._validate()
        X, y = check_arrays(X, y, sparse_format="csc")
        n_samples, n_features = X.shape

        # TODO: ensure only discrete features

        if self.k is None:
            k = n_features // 2
        else:
            k = self.k

        X_classes = np.array(list(set(X.reshape((n_samples * n_features,)))))
        y_classes = np.array(list(set(y.reshape((n_samples,)))))

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
