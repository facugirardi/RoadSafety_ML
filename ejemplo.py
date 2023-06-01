from sklearn import datasets
from xgboost import XGBClassifier
import shap

# Lectura de datos
data = datasets.load_breast_cancer()
print(data['DESCR'])
print(data.target_names)

# Datos necesarios para realizar las predicciones
x = data['data']

# Variable a predecir (prediccion)
y = data['target']

# Crear modelo
model = XGBClassifier()

model.fit(x, y)

""" Ya se puede empezar a realizar predicciones
 En este caso predict_proba arroja un vector en el que se muestra la probabilidad del cancer benigno y el maligno. 
"""

pred_prob = model.predict_proba(x)
print(pred_prob)

# Explicar el modelo utilizando SHAP
explainer_shap = shap.TreeExplainer(model)

shap_values = explainer_shap.shap_values(x)
shap.summary_plot(shap_values, x, feature_names=data['feature_names'])