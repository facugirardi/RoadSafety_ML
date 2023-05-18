from sklearn.model_selection import GridSearchCV, train_test_split
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

# Division de datos
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.40)

# Crear modelo
model = XGBClassifier()

# Definir parametros
parameters = {
}

gs = GridSearchCV(model, parameters)
gs.fit(X_test, y_test, verbose=False)

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

shap_values = explainer_shap.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=data['feature_names'])