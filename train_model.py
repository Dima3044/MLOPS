import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from mlflow.models import infer_signature
import os
import sys

def train_ridge_grid():
    # 1. Загрузка подготовленных данных (выход download.py)
    df = pd.read_csv('df_clear.csv')
    
    # Фикс MLflow: целевая переменная должна быть float64 для корректной сигнатуры
    y = df['rings'].astype('float64')
    X = df.drop(columns=['rings'])
    
    # 2. Разделение и масштабирование
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    num_cols = ['length', 'diameter', 'height', 'whole weight', 'shucked weight', 
                'viscera weight', 'shell weight', 'shell ratio', 'volume']
    
    sc = StandardScaler()
    X_train[num_cols] = sc.fit_transform(X_train[num_cols])
    X_test[num_cols] = sc.transform(X_test[num_cols])
    
    # 3. Настройка MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", os.path.join(os.getcwd(), "mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Abalone_Regression")
    
    # 4. GridSearchCV только для Ridge
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 5.0, 10.0],
        'solver': ['auto', 'cholesky', 'lsqr', 'saga'],
        'fit_intercept': [True, False]
    }
    
    with mlflow.start_run(run_name="Ridge_GridSearch"):
        grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        
        # 5. Оценка
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # 6. Логирование в MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("test_mae", mae)
        
        # Фикс сигнатуры: явное приведение к float64 убирает warning MLflow
        signature = infer_signature(X_train.astype('float64'), y_pred.astype('float64'))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        print(f"Лучшие параметры Ridge: {best_params}")
        print(f"Test R2: {r2:.4f} | Test MAE: {mae:.4f}")
        
        # 7. Экспорт пути лучшей модели для деплоя
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        with open("best_model.txt", "w") as f:
            f.write(model_uri)
        print(f"Путь модели сохранён в best_model.txt: {model_uri}")

if __name__ == "__main__":
    try:
        train_ridge_grid()
    except Exception as e:
        print(f"Ошибка обучения: {e}")
        sys.exit(1)