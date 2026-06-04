import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split

from ftir_pred.config import RANDOM_SEED


def split_by_person(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split so that all samples from the same person land in the same partition.
    Returns X_train, X_test, y_train, y_test, groups_train.
    """
    unique_persons = np.unique(groups)
    persons_train, persons_test = train_test_split(
        unique_persons, test_size=test_size, random_state=random_state
    )
    mask_train = np.isin(groups, persons_train)
    mask_test = np.isin(groups, persons_test)
    return (
        X[mask_train], X[mask_test],
        y[mask_train], y[mask_test],
        groups[mask_train],
    )


def make_cv_splitter(n_splits: int = 5) -> GroupKFold:
    """GroupKFold ensures no person appears in both train and validation folds."""
    return GroupKFold(n_splits=n_splits)
