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
# The following tutorial is partly from [Increase citations, ease review & foster collaboration](https://ml.recipes) book by [Jesper Dramsch](https://ml.recipes).

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Let's have the data loaded and briefly look at the data.

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
penguins.head()

# %% [markdown]
# It looks like there is some missing values like in the row number 4 (and maybe more), which a little data cleaning can take care of.

# %%
penguins = penguins.dropna(axis='rows')
penguins.head()

# %% [markdown]
# Making references to different selected groups of features and the target variable (to be classified) will help and be reused down the line.

# %%
num_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
cat_features = ["sex"]
features = num_features + cat_features
target = ["species"]
penguins = penguins[features+target]
penguins

# %% [markdown]
# ## Exploratory Data Analysis
# Let's have a look into the data in details with statistics and visualizations.

# %%
penguins.describe()

# %%
# penguins[num_features].hist(bins=30, figsize=(15,10))

# %%
import seaborn as sns

pairplot_figure = sns.pairplot(penguins[num_features+['species']], hue="species")

# %% [markdown]
# The complete pairwise plots of our features show that there are 2-feature pairs between `bill_length_mm` and any of `bill_depth_mm`, `flipper_length_mm`, and `body_mass_g` being able to separate 3 species the best.
#
# We will firstly make a classification model with 2 features for the sake of visualization. The selected features are `bill_length_mm` and `flipper_length_mm`.

# %%
selected_features = ['bill_length_mm', 'flipper_length_mm']
#feel free to experiment with other combinations.

# %% [markdown]
# ## Data Splitting
# Every supervised machine learning model needs a dataset to train on to be able to predict unseen data and that data splitting comes into play. We split the whole dataset into a training set, generally a bigger portion, and the rest as a testing set to act as unseen data for model evaluation.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(penguins[selected_features], penguins[target[0]], random_state=20, train_size=.7)
X_train.head()

# %%
y_train.head()

# %%
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %% [markdown]
# ## Model Training
# Now, let's build a classification model with Support-Vector Machine and train it with the training data set.

# %%
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)

# %% [markdown]
# Okay, we get a decent accuracy score on the training data.

# %%
svm.score(X_train, y_train)

# %% [markdown]
# To see how well our model predicts unseen data, we evaluate the model against the test set.

# %%
svm.score(X_test, y_test)

# %% [markdown]
# Let's see how our SVM model learn to distinguish the (training) data from 3 species visually.

# %%
import matplotlib.patches

def plot_svm2d(X, f1, f2, y, svm):
    # create a predicted mesh
    s = 0.2
    x_min, x_max = X[f1].min() - 1, X[f1].max() + 1
    y_min, y_max = X[f2].min() - 1, X[f2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, s), np.arange(y_min, y_max, s))
    df = {f1: xx.ravel(),f2: yy.ravel()}
    xy = pd.DataFrame(df)
    Z = svm.predict(xy)
    fig, ax = plt.subplots()
    
    # Fill the plot with predicted mesh
    levels, categories = pd.factorize(Z, sort=True)
    levels = levels.reshape(xx.shape)
    ax.contourf(xx, yy, levels, cmap=plt.cm.coolwarm, alpha=0.3)

    # data scatter plot
    levels, categories = pd.factorize(y, sort=True)
    handles = [matplotlib.patches.Patch(color=plt.cm.coolwarm.resampled(3)(i), label=c) for i, c in enumerate(categories)]
    ax.scatter(X[f1], X[f2], c=levels, cmap=plt.cm.coolwarm, edgecolors='black')
    ax.set_xlabel('bill_length_mm')
    ax.set_ylabel('flipper_length_mm')
    ax.set_title("SVM")
    ax.legend(handles=handles, title='Species')
    plt.show()

plot_svm2d(X_train, selected_features[0], selected_features[1], y_train, svm)

# %% [markdown]
# As you can see, our SVM misclassifies a lot of Chinstrap penquins in the train set. Something we can do here is to scale our numerical features. This is crucial for many machine learning algorithms, as they can be sensitive to the scale of the input features.
#
# It can be observed from the plot above that the range of `flipper_length_mm` covering the data is twice as wide as that of `bill_length_mm`. You can scale these features into an equal range with `StandardScaler`. With `Pipeline` introduced here, you can integrate pre-processors and a machine learning model into a convenient workflow with possibilities beyond this example.

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

num_scaler = StandardScaler()

model = Pipeline(steps=[
    ('scaler', num_scaler),
    ('classifier', SVC())
])
model

# %% [markdown]
# Now, it's time to train and see the result from our improved model.

# %%
model.fit(X_train, y_train)
print(f'Training set Accuracy: {model.score(X_train, y_train)}')
print(f'Testing set Accuracy: {model.score(X_test, y_test)}')
plot_svm2d(X_train, selected_features[0], selected_features[1], y_train, model)

# %% [markdown]
# Our model now generalizes much better with a simple scaling trick!

# %% [markdown]
# ## Model 

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
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
    X, y = penguins[selected], penguins[target[0]]
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
