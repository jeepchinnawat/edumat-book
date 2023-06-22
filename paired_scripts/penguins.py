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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Palmer Penguins
#
# This notebook uses [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/) {cite:p}`palmerpenguins`. The dataset in CSV format used is downloaded from [here](https://gist.github.com/slopp/ce3b90b9168f2f921784de84fa445651).
#
# The following tutorial is from [Increase citations, ease review & foster collaboration](https://ml.recipes) book by [Jesper Dramsch](https://ml.recipes).

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Data Preparation

# %%
from pathlib import Path
import pandas as pd
import numpy as np

DATA_FOLDER = Path("..") / "data"
DATA_FILEPATH = DATA_FOLDER / "penguins.csv"
# Execute on cloud platform? (e.g. colab), try this path instead
# DATA_FILEPATH = "https://raw.githubusercontent.com/jeepchinnawat/edumat-book/main/data/penguins.csv"

penguins = pd.read_csv(DATA_FILEPATH)
penguins.info()

# %%
penguins

# %%
penguins_prep = penguins.dropna(axis='rows')

num_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
cat_features = ["sex"]
features = num_features + cat_features
target = ["species"]

# %% [markdown]
# ## Exploratory Data Analysis

# %%
penguins_prep.describe()

# %%
penguins_prep[num_features].hist(bins=50, figsize=(15,10))

# %%
import seaborn as sns

pairplot_figure = sns.pairplot(penguins_prep[num_features+['species']], hue="species")

# %%
penguins_final = penguins_prep[features+target]
penguins_final

# DATA_CLEAN_FILEPATH = DATA_FOLDER / "penguins_final.csv"
# penguins_final.to_csv(DATA_CLEAN_FILEPATH, index=False)

# %% [markdown]
# ## Model 

# %%
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import accuracy_score as acc
from sklearn.utils import resample

from ipywidgets import interact
import ipywidgets as widgets

def svm_interact(sFeatures, trainsize, mislabel):
    status_widget.value = 'Calculating...'
    selected = np.array(sFeatures)
    X, y = penguins_final[selected], penguins_final[target[0]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainsize, random_state=21, stratify=y)

    if mislabel > 0.0:
        y_mis = resample(y_train, n_samples=int(y_train.shape[0]*mislabel), replace=False, random_state=21)
        for i in y_mis.index:
            label=['Chinstrap','Gentoo','Adelie']
            label.remove(y_mis[i])
            newlabel = resample(label, n_samples=1)[0]
            y_train[i] = newlabel

    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    if 'sex' in selected:
        transformers=[
            ('num', num_transformer, selected[:-1]),
            ('cat', cat_transformer, selected[-1:])
        ]
    else:
        transformers=[
            ('num', num_transformer, selected)
        ]
        
    preprocessor = ColumnTransformer(transformers=transformers)
    
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC()),
    ])

    mcc_scorer = make_scorer(mcc)
    acc_scorer = make_scorer(acc)
    scores = cross_validate(model, X_train, y_train, cv=5,
                            scoring={"MCC": mcc_scorer, "ACC": acc_scorer})

    print("Cross Validation on training set")
    for k, v in scores.items():
        print(k, " : ", v)
    print("Avg ACC in CV: ", np.average(scores["test_ACC"]))
    print("Avg MCC in CV: ", np.average(scores["test_MCC"]))
    print()

    # model = model.fit(X_train, y_train)
    # print("Evaluation on test set")
    # print("ACC: ", acc_scorer(model, X_test, y_test))
    # print("MCC: ", mcc_scorer(model, X_test, y_test))
    print(X_train.shape, y_train.shape)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X_train, y_train, train_size=0.8, random_state=21, stratify=y_train)
    model = model.fit(X2_train, y2_train)
    print("Evaluation on test set")
    print("ACC: ", acc_scorer(model, X2_test, y2_test))
    print("MCC: ", mcc_scorer(model, X2_test, y2_test))
    
    
    status_widget.value = 'Calculation completed!'

features_widget = widgets.SelectMultiple(
    options=features,
    value=[features[0], features[1]],
    description='Features:',
    disabled=False
)
trainsize_widget = widgets.FloatSlider(
    value=0.7,
    min=0.5,
    max=0.8,
    step=0.05,
    description='%train data:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.0%',
)
mislabel_widget = widgets.FloatSlider(
    value=0.0,
    min=0.0,
    max=0.5,
    step=0.005,
    description='Mislabeled:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1%',
)
status_widget = widgets.Label(value='')

interact(svm_interact, sFeatures=features_widget, trainsize=trainsize_widget, mislabel=mislabel_widget)
display(status_widget)
