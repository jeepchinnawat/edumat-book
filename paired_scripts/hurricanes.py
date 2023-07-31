# ---
# jupyter:
#   jupytext:
#     formats: book//ipynb,paired_scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: edumat
#     language: python
#     name: edumat
# ---

# %% [markdown]
# # (outlined) North Atlantic Hurricanes

# %%
from pathlib import Path
import pandas as pd
import numpy as np

DATA_FOLDER = Path("..") / "data"
DATA_FILEPATH = DATA_FOLDER / "hurricanes.csv"
# Execute on cloud platform? (e.g. colab), try this path instead
# DATA_FILEPATH = "https://raw.githubusercontent.com/jeepchinnawat/edumat-book/main/data/hurricanes.csv"

hurricanes = pd.read_csv(DATA_FILEPATH)
hurricanes

# %% [markdown]
# ## Introductory Data Inspection

# %%
hurricanes.info()

# %%
features = ['FirstLat','FirstLon','MaxLat','MaxLon','LastLat','LastLon','MaxInt']
label = ['Type']

# %%
hurricanes[features].describe()

# %%
hurricanes[features].boxplot()

# %%
# hurricanes[features].hist(bins=30, figsize=(15,10))

# %% [markdown]
# ## Feature Selection

# %% [markdown]
# ### Univariate Selection

# %%
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

selector = SelectKBest(f_classif, k=2)
selector.fit(hurricanes[features], hurricanes[label[0]])
scores = selector.scores_

features_scores = pd.Series(scores, index=features)

fig, ax = plt.subplots()
features_scores.plot.barh(ax=ax, color='skyblue')
ax.grid(True, which='both', color='grey', linewidth=0.3)
ax.set_title("Univariate selection toward label 'Type'")
ax.set_xlabel("score")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Random Forest's Feature Importance

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Parameters for GridSearch
param_grid = {
'n_estimators': [10, 20, 30, 40],
'max_depth': [2,4,6, 8],
'min_samples_split': [2, 4],
    'max_features': [1,2,4,7]
}

# rf = RandomForestClassifier(random_state=20, n_jobs=2,min_samples_leaf=1)
# # Grid search with cross-validation
# cv_rf = GridSearchCV(estimator=rf, param_grid=param_grid)
# cv_rf.fit(X_train, y_train)
# print(f"Best parameters: {cv_rf.best_params_}")

# %%
rf = RandomForestClassifier(random_state=20, n_jobs=2, min_samples_leaf=1,
                            max_depth=4, max_features=2, min_samples_split=2, n_estimators=40)
rf.fit(hurricanes[features], hurricanes[label[0]])

importances = rf.feature_importances_
features_scores = pd.Series(importances, index=features)

fig, ax = plt.subplots()
features_scores.plot.barh(ax=ax, color='skyblue')
ax.grid(True, which='both', color='grey', linewidth=0.3)
ax.set_title("RF Feature importances")
ax.set_xlabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()

# %%
selected_features = ["FirstLat","MaxLat"]

# %% [markdown]
# ## Stratified Data Splitting

# %%
hurricanes.groupby('Type').Type.count().plot(kind='bar')

# %%
from sklearn.model_selection import train_test_split

X,y = hurricanes[selected_features].copy(), hurricanes[label[0]].copy()
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=.7, random_state=20, stratify=y)

# %% [markdown]
# ## Classification Model

# %%

# %% [markdown]
# ### Classification Report

# %% [markdown]
# ### Choices of Evaluation Metrics

# %% [markdown]
# ## Next...?
# - Re-classify the dataset with the model to have a simulated dataset (perfectly separable)
# - Adding uncertainty(noise) to features of the simulated dataset
# - Predicted a noisy dataset and compare to the simulated
# - Expected: Affecting data points in the vicinity of decision boundaries

# %%
# tropical_hurricanes = hurricanes.copy()
# # set type 3 to type 1 for binary classification problem
# tropical_hurricanes[tropical_hurricanes['Type'] == 3] = 1
# # check that there are only type 0 and type 1
# tropical_hurricanes['Type'].unique()

# %%
# tropical_hurricanes['Type'].groupby(tropical_hurricanes['Type']).count()
