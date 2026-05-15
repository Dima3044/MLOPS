import pandas as pd
import numpy as np
import requests
import sys
from io import StringIO

def download_and_preprocess():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    cols = ["sex", "length", "diameter", "height", "whole weight", 
            "shucked weight", "viscera weight", "shell weight", "rings"]

    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text), names=cols)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    # Очистка выбросов (IQR + эмпирические границы)
    Q1, Q3 = df['height'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    df = df[(df['height'] <= 0.250) & (df['height'] >= lower_bound) & (df['rings'] <= 22)]

    # Feature Engineering
    df['shell ratio'] = df['shell weight'] / (df['whole weight'] + 1e-6)
    df['volume'] = df['length'] * df['diameter'] * df['height']
    
    # Квадратичный корень для линейности
    weight_cols = ['whole weight', 'shucked weight', 'viscera weight', 'shell weight', 'volume']
    for col in weight_cols:
        df[col] = np.sqrt(df[col])

    # One-Hot Encoding для sex
    sex_dummies = pd.get_dummies(df['sex'], prefix='sex', drop_first=True, dtype=int)
    df = pd.concat([df.drop('sex', axis=1), sex_dummies], axis=1)

    df.to_csv('df_clear.csv', index=False)
    print("df_clear.csv успешно создан.")

if __name__ == "__main__":
    download_and_preprocess()