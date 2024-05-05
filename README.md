# Presentación SHAP


### Indice
- [Resumen](https://github.com/facugirardi/shap-prog3/blob/main/README.md#resumen)
- [Problematica](https://github.com/facugirardi/shap-prog3/blob/main/README.md#problematica)
- [Instalación](https://github.com/facugirardi/shap-prog3/blob/main/README.md#instalación)
- [Ejemplo de Uso](https://github.com/facugirardi/shap-prog3/blob/main/README.md#ejemplo-de-uso)
- [Recursos](https://github.com/facugirardi/shap-prog3/blob/main/README.md#recursos)
- [Developers](https://github.com/facugirardi/shap-prog3/blob/main/README.md#developers)


### Resumen
SHAP es una librería diseñada para implementar Inteligencia Artificial Explicable (XAI, por sus siglas en inglés, eXplainable Artificial Intelligence). Utiliza conceptos de la teoría de juegos para determinar qué variables tienen mayor influencia en las predicciones de técnicas de aprendizaje automático. 
En esta aplicación específica, el propósito es calcular la probabilidad de sufrir un accidente automovilístico utilizando datos como temperatura, velocidad del viento, humedad, entre otros. Además, el proyecto incluye una integración de la API de OpenAI para un ChatBot relacionado con la seguridad vial.


### Problematica
La Inteligencia Artificial Explicable es una herramienta crucial en el aprendizaje automático para comprender modelos opacos o las denominadas "cajas negras". 
Dado el elevado número de accidentes de tráfico diarios, este proyecto busca prevenirlos y facilitar la comprensión de las razones detrás de ellos. Además, cuenta con un ChatBot especializado en seguridad vial que permite conversar, hacer preguntas y recibir consejos.


### Instalación
> - pip install shap==0.41
> - pip install xgboost==1.7.5
> - pip install streamlit==1.22
> - pip install pandas==1.3.5
> - pip install openai==0.27.7


### Ejecución Ejemplo de Uso
> - python -m streamlit run main.py
