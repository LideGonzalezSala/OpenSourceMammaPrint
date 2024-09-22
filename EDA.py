# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 19:20:33 2024

@author: lideg
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer
import numpy as np
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest

# Obtener el archivo desde el repositorio de github
url = 'https://raw.githubusercontent.com/LideGonzalezSala/OpenSourceMammaPrint/main/tcga_brca_risk.tsv'

# Cargar el dataset
df = pd.read_csv(url, sep='\t')

# Mostrar tamaño del dataset
print(f'Tamaño del dataset: {df.shape}')

# Mostrar primeras filas del dataset
print('Vista preliminar del dataset:')
print(df.head())

# Adaptación de nombres de columnas
df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_').str.replace('.', '_')

# Convertir la variable de riesgo binario a numérica para análisis
df['risk_binary_numeric'] = df['risk_binary'].map({'Bajo.riesgo': 0, 'Alto.riesgo': 1})

# Crear una carpeta para guardar las imágenes
if not os.path.exists('EDA_output'):
    os.makedirs('EDA_output')

x = df.drop(columns=["sample_name", "risk_binary", "risk_score","risk_binary_numeric"])
y = df["risk_binary_numeric"]

# Estadísticas Descriptivas
desc_stats = df.describe()
desc_stats.to_csv('EDA_output/descriptive_statistics.csv')

# Hacemos que el texto de las gráficas se vea un poco más grande
sns.set_context("notebook", font_scale=1.1)

# Distribución de la variable risk_binary
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='risk_binary')
plt.title('Distribución de la variable risk_binary')
plt.xlabel('risk_binary')
plt.ylabel('Frecuencia')

plt.savefig('EDA_output/risk_binary_distribution.svg')
plt.show()
plt.close()

# Distribución de risk_score
plt.figure(figsize=(10, 6))
sns.histplot(df['risk_score'], bins=30, kde=True)
plt.title('Distribución de la variable continua risk_score')
plt.xlabel('risk_score')
plt.ylabel('Frecuencia')

plt.savefig('EDA_output/risk_score_distribution.svg')
plt.show()
plt.close()

df[x.columns] = (
    FunctionTransformer(np.log1p)
    .set_output(transform="pandas")
    .fit_transform(df[x.columns])
)
df[x.columns] = (
    QuantileTransformer().set_output(transform="pandas").fit_transform(df[x.columns])
)

# Calcular la correlación de Pearson entre cada gen y la variable risk_binary_numeric
numeric_df = df.drop(columns=['sample_name', 'risk_binary', 'risk_score'])
correlations = numeric_df.corr()['risk_binary_numeric'].abs()
top_genes = correlations.sort_values(ascending=False).head(21).index  # Incluímos 'risk_binary_numeric' y los 20 genes más correlacionados

# Mapa de Calor de la Correlación con Risk Binary
selected_columns = list(top_genes)
corr_matrix = df[selected_columns].corr()
plt.figure(figsize=(14, 10))
ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 8})
plt.title('Mapa de calor de los genes más correlacionados con risk_binary')

new_labels = [label if label != 'risk_binary_numeric' else 'risk_binary' for label in selected_columns]

ax.set_xticklabels(new_labels, rotation=45, ha='right')
ax.set_yticklabels(new_labels, rotation=0)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig('EDA_output/mapa_calor_genes.svg', bbox_inches='tight')
plt.show()
plt.close()

# Creamos lista a partir de el índice top_genes
top_genes_list = list(top_genes)

# Creamos una figura con una cuadrícula de 2x2
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Bucle para recorrer el top 4 de genes con mayor correlación y generar violinplots
for i in range(1, 5):
    fila = (i - 1) // 2
    columna = (i - 1) % 2
    
    sns.violinplot(x="risk_binary_numeric", y=top_genes_list[i], data=df, ax=axs[fila, columna])
    
    
    axs[fila, columna].set_xticks([0, 1])
    axs[fila, columna].set_xticklabels(['Bajo.riesgo', 'Alto.riesgo'])
    
    axs[fila, columna].set_title(f'TOP {i} gen correlacionado con risk_binary: {top_genes_list[i]}')   
    axs[fila, columna].set_ylabel(f'Expresión de {top_genes_list[i]}')  # Etiqueta del eje y
    axs[fila, columna].set(xlabel=None)

plt.tight_layout()

plt.savefig('EDA_output/violinplot_top_genes.svg', bbox_inches='tight')
plt.show()
plt.close()



fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111)

X_reduced = TSNE().fit_transform(
    SelectKBest(k=100).fit_transform(df[x.columns], df["risk_binary"])
)
pl = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=df["risk_score"],
    s=40,
    cmap="coolwarm",
    alpha=0.5,
)

ax.set_xlabel("TSNE 0")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("TSNE 1")
ax.yaxis.set_ticklabels([])

plt.title('Visualización TSNE por riesgo')

plt.colorbar(pl)
plt.savefig('EDA_output/TSNE.svg', bbox_inches='tight')
plt.show()
plt.close()
