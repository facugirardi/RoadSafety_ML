import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import datasets

#Graficos
import seaborn as sns
import matplotlib.pyplot as plt

#Modelo
from xgboost import XGBClassifier

#Explicar
import shap


# lectura de datos
data = datasets.load_breast_cancer()
print(data['DESCR'])

# Datos necesarios para realizar las predicciones
X = data['data']

# Variable a predecir (prediccion)
y = data['target']

# ENTRENAMIENTO DEL MODELO
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.50)

# Crear modelo
model = XGBClassifier()

parameters = {
    'learning_rate': [0.01, 0.1, 0.001],
    'max_depth': [3, 4, 5],
    'n_estimators': [10, 50, 100]
}

gs = GridSearchCV(model, parameters)
gs.fit(X_eval, y_eval, verbose=False)

# Seleccionar los mejores parametros
model.set_params(**gs.best_params_)
model.fit(X_train, y_train)

""" Ya se puede empezar a realizar predicciones
 En este caso predict_proba arroja un vector en el que se muestra la probabilidad del cancer benigno y el maligno. 
"""
pred_prob = model.predict_proba(X_train)
print(pred_prob)

# Explicar el modelo utilizando SHAP
explainer_shap = shap.TreeExplainer(model)

shap_values = explainer_shap.shap_values(X_eval)
shap.summary_plot(shap_values, X_eval, feature_names=data['feature_names'])