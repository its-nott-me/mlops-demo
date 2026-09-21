"""
Section 5.6 / 7 - Evidently AI drift & data-quality monitoring.

Run:
    python evidently_report.py
Then open drift_report.html in a browser and screenshot the dashboard.
"""
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

from data_prep import X_train, X_test

# Treat the training split as the "reference" data and the test split as
# "current" production data, to illustrate drift detection.
reference = X_train.copy()
current = X_test.copy()

report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
report.run(reference_data=reference, current_data=current)
report.save_html("drift_report.html")
print("Report saved -> open drift_report.html in your browser and screenshot it.")
