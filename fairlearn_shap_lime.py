"""
Section 5.7 / 7 - Governance: Fairlearn (fairness) + SHAP + LIME (explainability).

Run:
    python fairlearn_shap_lime.py
Screenshot: the printed accuracy-by-group table, the SHAP bar plot window,
and lime_explanation.html opened in a browser.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import shap
from lime.lime_tabular import LimeTabularExplainer

from data_prep import X_train, X_test, y_train, y_test, A_train, A_test

# 1. Train a baseline model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 2. Fairness metrics, broken down by the sensitive feature (Sex)
mf = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=y_pred, sensitive_features=A_test)
print("Accuracy by group:\n", mf.by_group)
print(
    "Demographic parity difference:",
    demographic_parity_difference(y_test, y_pred, sensitive_features=A_test),
)
print(
    "Equalized odds difference:",
    equalized_odds_difference(y_test, y_pred, sensitive_features=A_test),
)

# 3. Mitigate unfairness with Fairlearn's ExponentiatedGradient reduction
mitigator = ExponentiatedGradient(
    LogisticRegression(max_iter=1000), constraints=DemographicParity()
)
mitigator.fit(X_train, y_train, sensitive_features=A_train)
y_pred_mitigated = mitigator.predict(X_test)
print(
    "Demographic parity AFTER mitigation:",
    demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=A_test),
)

# 4. SHAP - global + local feature importance
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.plots.bar(shap_values, show=True)            # global feature importance
shap.plots.waterfall(shap_values[0], show=True)   # explains a single prediction

# 5. LIME - local explanation for one prediction
lime_explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["<=50K", ">50K"],
    mode="classification",
)
exp = lime_explainer.explain_instance(
    X_test.iloc[0].values, model.predict_proba, num_features=8
)
exp.save_to_file("lime_explanation.html")
print("Saved LIME explanation -> open lime_explanation.html in your browser.")
