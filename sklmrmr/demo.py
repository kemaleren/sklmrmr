"""Run MRMR on a datasets."""

import csv
import os.path
import sys
from time import time
from argparse import ArgumentParser, FileType

import numpy as np
from sklearn.datasets import load_digits

from sklmrmr import MRMR


DEFAULT_FILE = os.path.join(os.path.dirname(__file__),
                            'test_nci9_s3.csv')


def read_csv(handle):
    rdr = csv.reader(handle)
    rows = list(row for row in rdr)
    raw_data = np.vstack(list(list(map(int, row)) for row in rows[1:]))
    if handle != sys.stdin:
        handle.close()
    y = raw_data[:, 0]
    X = raw_data[:, 1:]
    return X, y


def get_digits():
    digits = load_digits()
    X = digits.images.reshape((len(digits.images), -1)).astype(int)
    y = digits.target
    return X, y


if __name__ == '__main__':
    args = sys.argv[1:]
    parser = ArgumentParser(description='minimum redundancy'
                            ' maximum relevance feature selection')
    parser.add_argument('--method', type=str, default="mid")
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--n_features', type=int, default=10)
    parser.add_argument('--file', type=FileType('r'),
                        default=open(DEFAULT_FILE))
    parser.add_argument('--digits', action='store_true')
    ns = parser.parse_args(args)

    if ns.digits:
        X, y = get_digits()
        data_name = 'digits'
    else:
        X, y = read_csv(ns.file)
        data_name = ns.file.name

    model = MRMR(k=ns.n_features, method=ns.method, normalize=ns.normalize)
    names = list("feature_{}".format(i) for i in range(X.shape[1]))

    print('running on {}'.format(data_name))
    print('model: {}'.format(model))

    t = time()
    model.fit(X, y)
    t = time() - t
    print("time: {:.3f} seconds".format(t))

    selected_names = list(names[i]
                          for i in np.argsort(model.ranking_)[:model.k])
    print("selected features:\n{}".format(", ".join(selected_names)))
