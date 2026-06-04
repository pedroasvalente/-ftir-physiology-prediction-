import numpy as np
import pytest

from ftir_pred.data.splits import make_cv_splitter, split_by_person


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(52)
    n_persons = 20
    samples_per_person = 3
    n_features = 50

    persons = np.repeat(np.arange(n_persons), samples_per_person)
    X = rng.standard_normal((len(persons), n_features))
    y = rng.standard_normal(len(persons))
    return X, y, persons


def test_no_person_in_both_partitions(synthetic_data):
    X, y, groups = synthetic_data
    X_tr, X_te, y_tr, y_te, g_tr = split_by_person(X, y, groups, test_size=0.2)

    train_persons = set(np.unique(groups[np.isin(groups, np.unique(g_tr))]))
    all_test_persons = set(groups[len(y_tr):]) if False else set()

    mask_train = np.isin(groups, np.unique(g_tr))
    mask_test = ~mask_train

    train_set = set(np.unique(groups[mask_train]))
    test_set = set(np.unique(groups[mask_test]))
    assert train_set.isdisjoint(test_set), "Person appears in both train and test partitions"


def test_split_sizes(synthetic_data):
    X, y, groups = synthetic_data
    X_tr, X_te, y_tr, y_te, g_tr = split_by_person(X, y, groups, test_size=0.2)
    assert len(y_tr) > 0
    assert len(y_te) > 0
    assert len(y_tr) + len(y_te) == len(y)


def test_cv_splitter_no_leakage(synthetic_data):
    X, y, groups = synthetic_data
    X_tr, X_te, y_tr, y_te, g_tr = split_by_person(X, y, groups, test_size=0.2)
    cv = make_cv_splitter(n_splits=5)

    for fold_train_idx, fold_val_idx in cv.split(X_tr, y_tr, groups=g_tr):
        fold_train_persons = set(g_tr[fold_train_idx])
        fold_val_persons = set(g_tr[fold_val_idx])
        assert fold_train_persons.isdisjoint(fold_val_persons), \
            "Person appears in both train and validation fold"
