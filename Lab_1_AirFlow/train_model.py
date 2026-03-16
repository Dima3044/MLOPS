import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train():
    df = pd.read_csv("./df_clear.csv")
    X = df.drop('rings', axis=1).values
    Y = df['rings']
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.3, random_state=42)

    params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    mlflow.set_experiment("random forest rings")
    with mlflow.start_run():
        rf = RandomForestRegressor(random_state=42)

        clf = GridSearchCV(rf, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train)

        best = clf.best_estimator_
        y_pred = best.predict(X_val)

        (rmse, mae, r2) = eval_metrics(y_val, y_pred)

        mlflow.log_param("n_estimators", best.n_estimators)
        mlflow.log_param("max_depth", best.max_depth)
        mlflow.log_param("min_samples_split", best.min_samples_split)
        mlflow.log_param("min_samples_leaf", best.min_samples_leaf)
        mlflow.log_param("bootstrap", best.bootstrap)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)

        with open("rf_rings.pkl", "wb") as file:
            joblib.dump(best, file)
