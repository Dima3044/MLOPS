import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import requests
from pathlib import Path
import os
from datetime import timedelta
from train_model import train


def calc_IQR(df, name, coef=1.5):
    y = df[name]
    Q1 = np.quantile(y, 0.25)
    Q3 = np.quantile(y, 0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + coef * IQR
    lower_bound = Q1 - coef * IQR
    return upper_bound, lower_bound

def download_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"

    column_names = ["sex", "length", "diameter", "height", "whole weight", 
                    "shucked weight", "viscera weight", "shell weight", "rings"]
    df = pd.read_csv(url, names=column_names)
    df.to_csv('abalones_df.csv', index=False)
    print("df:", df.shape)
    return df

def clear_data():
    column_names = ["sex", "length", "diameter", "height", "whole weight", 
                    "shucked weight", "viscera weight", "shell weight", "rings"]

    df = pd.read_csv('abalones_df.csv')

    # Верхняя и нижняя граница интерквартильным размахом
    upper_bound, lower_bound = calc_IQR(df, 'height')

    # Очистка подозрительно больших значений
    df = df[df['height'] <= 0.250]

    # Очистка подозрительно малых значений
    df = df[df['height'] >= lower_bound]

    # Удаляем слишком старых моллюсков
    df = df[df['rings'] <= 22]

    # Отбор признаков
    df = df[['height', 'shell weight', 'sex', 'rings']]

    # Weight признаки к более линейному виду
    df['shell weight'] = np.sqrt(df['shell weight'])

    sex_dummies = pd.get_dummies(df['sex'], dtype=int, drop_first=True)
    df = pd.concat([df, sex_dummies], axis=1)
    df = df.drop('sex', axis=1)

    df.to_csv('df_clear.csv', index=False)
    return True


dag_abalones = DAG(
    dag_id="train_pipe",
    start_date=datetime(2026, 3, 15),
    schedule=timedelta(minutes=5),
    max_active_runs=1,
    catchup=False
)

download_task = PythonOperator(python_callable=download_data, task_id = "download_abalones", dag = dag_abalones)
clear_task = PythonOperator(python_callable=clear_data, task_id = "clear_abalones", dag = dag_abalones)
train_task = PythonOperator(python_callable=train, task_id = "train_abalones", dag = dag_abalones)
download_task >> clear_task >> train_task