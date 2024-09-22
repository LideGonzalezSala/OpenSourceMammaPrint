# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:35:17 2024

@author: lideg
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFdr
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import os
from sklearn.feature_selection import VarianceThreshold


# Obtener el archivo desde el repositorio de github
url = 'https://raw.githubusercontent.com/LideGonzalezSala/OpenSourceMammaPrint/main/tcga_brca_risk.tsv'

# Cargar el dataset
data = pd.read_csv(url, sep='\t')

# Preprocesamiento del dataset
print("Preprocesando los datos...")
data = data.drop(columns=['risk_score'])
data['risk_binary'] = data['risk_binary'].apply(lambda x: 1 if x == 'Alto.riesgo' else 0)
data.columns = data.columns.str.replace(' ', '_').str.replace('-', '_').str.replace('.', '_')

X = data.drop(columns=['sample_name', 'risk_binary'])
y = data['risk_binary']

# Separar en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

# Pipeline
pipeline = Pipeline([
    ('log', FunctionTransformer(np.log1p)),
    ('variance', VarianceThreshold()),
    ('scaler', QuantileTransformer(random_state=0)),
    ('selector', SelectFdr()),
    ('classifier', RidgeClassifierCV())
])

# Ajustar el modelo final y obtener las mejores métricas
print("Ajustando el modelo final con el conjunto de entrenamiento...")
pipeline.fit(X_train, y_train)

# Predicciones y evaluación final en el conjunto de prueba
print("Realizando predicciones y evaluando el modelo final en el conjunto de prueba...")
y_prob = pipeline.decision_function(X_test)

# Convertir probabilidades continuas en predicciones binarias usando un umbral de 0.5
y_pred = (y_prob > 0.5).astype(int)

# Crear una carpeta para guardar las imágenes y archivos
if not os.path.exists('Train-Test_Split_output'):
    os.makedirs('Train-Test_Split_output')

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)  # Mostrar la matriz de confusión en pantalla
conf_matrix_df = pd.DataFrame(conf_matrix, index=['REAL Bajo.riesgo', 'REAL Alto.riesgo'], columns=['PREDICCION Bajo.riesgo', 'PREDICCION Alto.riesgo'])
conf_matrix_df.to_csv('Train-Test_Split_output/confusion_matrix.csv')

# Reporte de clasificación
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)  # Mostrar el reporte de clasificación en pantalla
with open('Train-Test_Split_output/classification_report.txt', 'w') as f:
    f.write(class_report)

# Curva ROC
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, label='Curva ROC (área = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de FALSOS POSITIVOS')
plt.ylabel('Tasa de VERDADEROS POSITIVOS')
plt.title('Curva ROC de partición aleatoria Train-Test')
plt.legend(loc="lower right")

plt.savefig('Train-Test_Split_output/ROC.svg')
plt.show()
plt.close()


