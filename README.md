# SHAP Presentation


### Table of Contents
- [Summary](#Summary)
- [Problem Statement](#Problem-Statement)
- [Installation](#Installation)
- [Example Usage](#Example-Usage)


### Summary
SHAP is a library designed to implement Explainable Artificial Intelligence (XAI). It utilizes concepts from game theory to determine which variables have the most significant influence on machine learning predictions. 
In this specific application, the goal is to calculate the probability of experiencing a car accident using data such as temperature, wind speed, humidity, among others. Additionally, the project integrates the OpenAI API for a ChatBot related to road safety.

### Problem Statement
Explainable Artificial Intelligence is a crucial tool in machine learning for understanding opaque models or so-called "black boxes". 
Given the high number of daily traffic accidents, this project aims to prevent them and facilitate understanding the underlying reasons. Moreover, it features a specialized ChatBot in road safety that enables conversation, questions, and advice.


### Installation
> - pip install shap==0.41
> - pip install xgboost==1.7.5
> - pip install streamlit==1.22
> - pip install pandas==1.3.5
> - pip install openai==0.27.7


### Example Usage
> - python -m streamlit run main.py
