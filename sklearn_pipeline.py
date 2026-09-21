"""
Section 5.3 - Scikit-learn model development with a Pipeline, cross-validation,
and hyperparameter search. Also saves the trained model to model.joblib so
main.py (FastAPI) can load and serve it.

Run:
    python sklearn_pipeline.py
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib

from data_prep import X_train, y_train, X_test, y_test

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])

scores = cross_val_score(pipe, X_train, y_train, cv=5)
print("5-fold CV accuracy: %.4f (+/- %.4f)" % (scores.mean(), scores.std()))

param_grid = {"clf__C": [0.01, 0.1, 1, 10]}
search = GridSearchCV(pipe, param_grid, cv=5)
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
print("Test accuracy with best model:", search.score(X_test, y_test))

joblib.dump(search.best_estimator_, "model.joblib")
print("Saved trained model to model.joblib")
