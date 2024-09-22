# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 20:26:35 2024

@author: lideg
"""

import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")
sns.set_context("paper", font_scale=2)

scores = joblib.load("scores.jbl")

models = scores["estimator"]

coef_lst = [
    pd.Series(model[-1].coef_.ravel(), index=model[-1].feature_names_in_)
    for model in models
]

coef_df = pd.concat(coef_lst, axis=1, join="inner").T
coef_df.head()

genes_to_keep = coef_df.apply(np.sign).apply(lambda x: len(set(x)))
genes_to_keep = genes_to_keep.index[genes_to_keep == 1]
genes_to_keep.size

genes_top30 = (
    coef_df[genes_to_keep].abs().mean().sort_values(ascending=False)[:30].index
)

genes_top30_sorted = coef_df[genes_top30].mean().sort_values(ascending=True).index
coef_df_top30 = coef_df[genes_top30_sorted].melt(
    var_name="gene", value_name="coefficient"
)

with sns.plotting_context("paper", font_scale=1):
    with sns.axes_style("whitegrid"):
        sbp = sns.boxplot(data=coef_df_top30, x="coefficient", y="gene", palette="vlag")
        sns.despine(top=True, left=True, right=True, bottom=True)

        sbp.set_xlabel(r"$\ell_{2}$ Coefficient score", fontsize=12)
        sbp.set_ylabel("Gene", fontsize=12)
        sbp.yaxis.grid(True)  # Hide the horizontal gridlines
        sbp.xaxis.grid(False)  # Show the vertical gridlines
        sbp.axvline(0, linewidth=2, color="black", linestyle="--")

        fig = plt.gcf()
        fig.set_size_inches(8, 8 / 1.6)
        plt.savefig("coefs.png", dpi=300, bbox_inches="tight")