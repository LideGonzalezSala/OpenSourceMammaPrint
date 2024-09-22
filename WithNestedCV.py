# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 21:57:45 2024

@author: lideg
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFdr
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import os

# Obtener el archivo desde el repositorio de github
url = 'https://raw.githubusercontent.com/LideGonzalezSala/OpenSourceMammaPrint/main/tcga_brca_risk.tsv'

# Cargar el dataset
data = pd.read_csv(url, sep='\t')

# Preprocesamiento del dataset
data = data.drop(columns=['risk_score'])
data['risk_binary'] = data['risk_binary'].apply(lambda x: 1 if x == 'Alto.riesgo' else 0)
data.columns = data.columns.str.replace(' ', '_').str.replace('-', '_').str.replace('.', '_')

X = data.drop(columns=['sample_name', 'risk_binary'])
y = data['risk_binary']

# Pipeline
pipeline = Pipeline([
    ('log', FunctionTransformer(np.log1p)),
    ('variance', VarianceThreshold()),
    ('scaler', QuantileTransformer(n_quantiles=10, random_state=0)),
    ('selector', SelectFdr()),
    ('classifier', RidgeClassifierCV())
])

cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)

scoring_system = {
    'roc_auc_score': 'roc_auc',
    'f1_score': 'f1',
    'precision_score': 'precision',
    'recall_score': 'recall',
    'accuracy_score': 'accuracy',
}

# Nested cross-validation
scores = cross_validate(estimator=pipeline, X=X, y=y, cv=cv, scoring=scoring_system, n_jobs=-1, return_train_score=True, return_estimator=True)

# Crear una carpeta para guardar los informes y figuras
if not os.path.exists('WithNestedCV_output'):
    os.makedirs('WithNestedCV_output')

# Guardar los resultados del cross-validation en un CSV
scores_df = pd.DataFrame(scores)
csv_path = os.path.join('WithNestedCV_output', 'scores.csv')
scores_df.to_csv(csv_path, index=False)

# Figura de distribución de métricas de rendimiento en el orden especificado
plt.figure(figsize=(8, 6))  # Adjust figure size if necessary
plt.boxplot([
    scores_df['test_precision_score'],   # Precision
    scores_df['test_recall_score'],      # Recall
    scores_df['test_f1_score'],          # F1
    scores_df['test_accuracy_score'],    # Accuracy
    scores_df['test_roc_auc_score'],     # ROC AUC
],
labels=['Precision', 'Recall', 'F1', 'Accuracy', 'ROC AUC'])

plt.ylabel('Score', fontsize=18)  # Even larger y-label font size
plt.title('Distribución de métricas de rendimiento', fontsize=20)  # Larger title font size
plt.xticks(rotation=45, fontsize=16)  # Larger x-tick labels
plt.yticks(fontsize=16)  # Larger y-tick labels

plt.tight_layout()
dist_svg_path = os.path.join('WithNestedCV_output', 'distr_metr_rendimiento.svg')
plt.savefig(dist_svg_path, format='svg', bbox_inches='tight')
plt.show()
