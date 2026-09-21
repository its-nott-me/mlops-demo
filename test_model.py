"""Simple smoke tests so the GitHub Actions CI workflow has something to run."""
from sklearn.linear_model import LogisticRegression

from data_prep import X_train, y_train, X_test, y_test


def test_model_trains_and_predicts():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)


def test_model_beats_random_guessing():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    assert acc > 0.5
