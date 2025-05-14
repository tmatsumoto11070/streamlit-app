# save_models.py
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# 原材料比率：0～1、温度：20～50、混練時間：0～60 → 正規化して学習
X_raw = np.random.rand(200, 4)
X_raw[:, 2] = X_raw[:, 2] * 30 + 20    # 温度 → 20～50
X_raw[:, 3] = X_raw[:, 3] * 60         # 時間 → 0～60

# 正規化してモデル学習（本番と一致させる）
X = np.copy(X_raw)
X[:, 2] = X[:, 2] / 50
X[:, 3] = X[:, 3] / 60

y_visc = X @ np.array([50, 30, 10, 5]) + 10
y_absorp = X @ np.array([40, 20, -15, 5]) + 5
y_str = X @ np.array([60, -10, 20, 15]) + 7

joblib.dump(LinearRegression().fit(X, y_visc), "models/model_viscosity.joblib")
joblib.dump(LinearRegression().fit(X, y_absorp), "models/model_absorption.joblib")
joblib.dump(LinearRegression().fit(X, y_str), "models/model_strength.joblib")
