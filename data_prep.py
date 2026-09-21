"""
Shared data preparation used by every script in this project.
All other scripts do `from data_prep import ...` so everyone trains/tests
on the exact same split - that's what makes the individual tool demos
(MLflow, scikit-learn, Fairlearn/SHAP/LIME, FastAPI, Evidently) consistent.

Dataset: the classic "Adult / Census Income" dataset, bundled with shap.
- X: features (already numeric-encoded) - Age, Workclass, Education-Num,
  Marital Status, Occupation, Relationship, Race, Sex, Capital Gain,
  Capital Loss, Hours per week, Country
- y: True/False, whether income is > $50K
- "Sex" is used as the sensitive feature for the Fairlearn section.
"""
import shap
from sklearn.model_selection import train_test_split

X, y = shap.datasets.adult()
A = X["Sex"]  # sensitive feature: 0 = Female, 1 = Male

X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X, y, A, test_size=0.3, random_state=42, stratify=y
)

if __name__ == "__main__":
    print("Train size:", X_train.shape, " Test size:", X_test.shape)
    print(X_train.head())
