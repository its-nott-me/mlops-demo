"""
Section 5.2 / 6 / 7 - MLflow experiment tracking.

Run:
    python mlflow_experiment.py
Then, in another terminal, from the same folder:
    mlflow ui
Open http://localhost:5000 and screenshot the run you see there.
"""
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_prep import X_train, X_test, y_train, y_test

mlflow.set_experiment("mlops-demo")
mlflow.sklearn.autolog()  # automatically logs params, metrics, and the model artifact

with mlflow.start_run(run_name="logistic-regression-baseline"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    # autolog already captures this, but logging it explicitly makes it
    # easy to find in the MLflow UI's metrics column
    mlflow.log_metric("test_accuracy", acc)

print("Run finished. Start 'mlflow ui' and open http://localhost:5000 to view it.")
