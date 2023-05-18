import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import streamlit as st
import shap

data = pd.read_csv('/home/facu/shap-prog3/shap-prog3/accident-data.csv')

X = data.drop('Probability', axis=1)
y = data['Probability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',  # Para problemas de clasificación binaria
    'eval_metric': 'logloss'  # Métrica de evaluación: log loss
}


num_rounds = 75  # Número de rondas de entrenamiento
model = xgb.train(params, dtrain, num_rounds)

explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap.shap_values(X_test)


def shap_enable():
    shap.summary_plot(shap_values, X_test, feature_names=names)


# Título de la app
st.title("Probabilidad de Accidente en Auto")

# Inputs en filas de a 3
row1_col1, row1_col2, row1_col3 = st.columns(3)
row2_col1, row2_col2, row2_col3 = st.columns(3)
row3_col1, row3_col2, row3_col3 = st.columns(3)

# Solicitar inputs
distancia = row1_col1.number_input("Distancia (KM)", min_value=0.0)
temperatura = row1_col2.number_input("Temperatura (Celsius)")
sensacion_termica = row1_col3.number_input("Sensación Térmica (Celsius)")
humedad = row2_col1.number_input("Humedad (%)", min_value=0, max_value=100, step=1)
presion = row2_col2.number_input("Presión (hPa)")
velocidad_viento = row2_col3.number_input("Velocidad del viento (km/h)")
precipitaciones = row3_col1.number_input("Precipitaciones (mm)")
year = row3_col2.number_input("Año", min_value=2000, max_value=2100, step=1)
mes = row3_col3.number_input("Mes", min_value=1, max_value=12, step=1)
hora = st.number_input("Hora", min_value=0, max_value=23, step=1)

# Conversion de medidas
distance = distancia / 1.6
temperature = (temperatura * 9/5) + 32
wind_chill = (sensacion_termica * 9/5) + 32
pressure = presion * 0.029530100
wind_speed = velocidad_viento / 1.6
precip = precipitaciones / 25.4


names = ['Distance(mi)',
        'Temperature(F)',
        'Wind_Chill(F)',
        'Humidity(%)',
        'Pressure(in)',
        'Visibility(mi)',
        'Wind_Speed(mph)',
        'Precipitation(in)',
        'Bump',
        'Crossing',
        'Roundabout',
        'Stop',
        'Traffic_Signal',
        'Accident_year',
        'Accident_month',
        'Accident_hour']


new_data = [
    {
        'Distance(mi)': distance,
        'Temperature(F)': temperature,
        'Wind_Chill(F)': sensacion_termica,
        'Humidity(%)': humedad,
        'Pressure(in)': pressure,
        'Visibility(mi)': 0.5,
        'Wind_Speed(mph)': wind_speed,
        'Precipitation(in)': precip,
        'Bump': True,
        'Crossing': False,
        'Roundabout': False,
        'Stop': True,
        'Traffic_Signal': True,
        'Accident_year': year,
        'Accident_month': mes,
        'Accident_hour': hora
    }
]

new_df = pd.DataFrame(new_data)

dnew = xgb.DMatrix(new_df)

y_new_pred_proba = model.predict(dnew)
print(y_new_pred_proba)

pred = y_new_pred_proba[0] * 100

prediccion_text = f'**Tenes {int(pred)}% de probabilidades de tener un accidente automovilistico.**'
prediccion_label = st.write(prediccion_text)

st.button("Activar SHAP", on_click=shap_enable())
