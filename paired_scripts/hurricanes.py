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
# The learning objectives of this notebook are to carry out feature selection for a classification model, train a classification model for types of hurricanes, and understand outcomes of feature uncertainty based on a experiment.
#
# The [North Atlantic Hurricanes dataset](https://myweb.fsu.edu/jelsner/temp/Data.html) used is developed by James B. Elsner and colleagues containing hurricanes recorded during the years 1944 to 2000. Each hurricane instance contains the year, the name (if it was named), the coordinates where it started, the last coordinates measured, the maximum coordinates (based on an aspect that an increment means closer to the coast), the maximum intensity, and the type. They are labeled into 3 types: tropical hurricanes (Type 0), hurricanes under baroclinic influences (Type 1), and hurricanes from baroclinic initiation (Type 3).
#
# Let's start with loading the data.

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# %% [markdown]
# We will leave out the year and go on with first, last, and maximum cooridinates and maximum intensity as our potential fields for classification task.

# %%
features = ['FirstLat','FirstLon','MaxLat','MaxLon','LastLat','LastLon','MaxInt']
label = ['Type']

# %% [markdown]
# We can look at simple statistics of our features with `describe` function of pandas' dataframe.

# %%
hurricanes[features].describe()

# %% [markdown]
# Simply replacing `describe` with `boxplot` function, we can visualize some of stats above with box plot.

# %%
hurricanes[features].boxplot()
plt.show()

# %% [markdown]
# ## Feature Selection
# In learning and working on machine learning or data science models in general, you sometimes have small datasets, but oftentimes you encounter with large datasets. They can be large by a big number of data instances or a number of features or both. However, a large number of features, also called a high dimensional dataset, is considered to be more critical to look at, because the larger it is, the bigger the feature space and the longer time learning algorithms take to find optimal solution to predict target variables. Also with irrelevant and unimportant features included, they can negatively impact models performance.
#
# The following sections demonstrate two feature selection techniques including univariate selection and Random Forrest's feature importances.

# %% [markdown]
# ### Univariate Selection
#
# Statistical tests can be used to check how well each feature discriminates between classes of a target (categorical) variable. To quantify such tests, We use scikit-learn's ANOVA (analysis of variance) F-value `f_classif` together with `SelectKBest` to select features according to the k highest scores, in our case 2.

# %%
from sklearn.feature_selection import SelectKBest, f_classif

f_test = SelectKBest(f_classif, k=2)
f_test.fit(hurricanes[features], hurricanes[label[0]])
scores = f_test.scores_

features_scores = pd.Series(scores, index=features)

fig, ax = plt.subplots()
features_scores.plot.barh(ax=ax, color='skyblue')
ax.grid(True, which='both', color='grey', linewidth=0.3)
ax.set_title("Univariate selection for label 'Type'")
ax.set_xlabel("score")
fig.tight_layout()
plt.show()

# %% [markdown]
# #### Intuition
# To understand how these statistical tests quantify features' discriminating scores, we plot the distributions of each hurricane type toward the feature with highest score `FirstLat` and the least one `FirstLon`.
#
# The graph shows that, even though the projected distributions onto `FirstLat` do not clearly discriminate the 3 classes, they do much better than those of `FirstLon` where all class distributions are completely overlapped onto each other.

# %%
import seaborn as sns

sns.jointplot(data=hurricanes, x="FirstLon", y="FirstLat", hue="Type", palette="tab10")
plt.show()

# %% [markdown]
# ### Random Forest's Feature Importance
# A Random Forest can be used to estimate the importances of features on a particular task. An importance of a feature is measured by looking at how much the tree nodes using that feature reduce impurity (Gini or Shannon information gain) across all trees in the forest on average.

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0, n_jobs=2, min_samples_leaf=1,
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
selected_features = ['FirstLat', 'MaxLat', 'MaxInt']

# %% [markdown]
# ## Why not FirstLat,MaxLat together?

# %%
import seaborn as sns

fig, axs = plt.subplots(1,2, figsize=(10, 4))

hurricanes.plot.scatter('FirstLat', 'MaxLat', c='Type', colormap='coolwarm_r', ax=axs[0])
axs[0].set_title("FirstLat, MaxLat")

# the abs. correlation matrix
df = pd.DataFrame(hurricanes, columns=selected_features)
abs_corr = df.corr().abs()
sns.heatmap(abs_corr, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=axs[1])
axs[1].set_title("Absolute Correlations")

plt.show()

# %%
selected_features = ['FirstLat', 'MaxInt']

# %% [markdown]
# ## Stratified Data Splitting

# %%
hurricanes.groupby('Type').Type.count().plot(kind='bar', color='skyblue')
plt.show()

# %%
from sklearn.model_selection import train_test_split

X,y = hurricanes[selected_features].copy(), hurricanes[label[0]].copy()
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=.7, random_state=0, stratify=y)

# %% [markdown]
# ## Classification Model

# %%
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

svm = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

svm.fit(X_train, y_train)
svm.score(X_train, y_train)

# %%
svm.score(X_test, y_test)

# %%
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

y_pred = svm.predict(X_test)

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation = 'vertical',
                                               cmap=plt.cm.Blues)
disp.figure_.suptitle("Confusion Matrix")

plt.show()

# %% [markdown]
# ### Transform to binary classification

# %%
tropical = hurricanes.copy()

# merge type 3 into type 1 for binary classification problem (tropical hurricane or not?)
tropical.loc[tropical['Type'] == 3, 'Type'] = 1
# check for binary type
tropical['Type'].unique()

# %%
X,y = tropical[selected_features].copy(), tropical[label[0]].copy()
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=.7, random_state=0, stratify=y)

# %%
import matplotlib.patches

def plot_2DClassifier(X, f, y, classifier, title):
    # create a predicted mesh
    s = 0.2
    f1, f2 = f[0], f[1]
    x_min, x_max = X[f1].min() - 1, X[f1].max() + 1
    y_min, y_max = X[f2].min() - 1, X[f2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, s), np.arange(y_min, y_max, s))
    df = {f1: xx.ravel(),f2: yy.ravel()}
    xy = pd.DataFrame(df)
    Z = classifier.predict(xy)
    fig, ax = plt.subplots()
    
    # Fill the plot with predicted mesh
    levels, categories = pd.factorize(Z, sort=True)
    levels = levels.reshape(xx.shape)
    ax.contourf(xx, yy, levels, cmap=plt.cm.coolwarm, alpha=0.3)

    # data scatter plot
    n_classes = classifier.classes_.shape[0]
    levels, categories = pd.factorize(y, sort=True)
    handles = [matplotlib.patches.Patch(color=plt.cm.coolwarm.resampled(n_classes)(i), label=c) for i, c in enumerate(categories)]
    ax.scatter(X[f1], X[f2], c=levels, cmap=plt.cm.coolwarm, edgecolors='black')
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title(title)
    ax.legend(handles=handles)
    plt.show()


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### SVM's hyperparameters (interactive w/ ipywidgets)

# %% editable=true slideshow={"slide_type": ""}
from ipywidgets import interact, fixed
import ipywidgets as widgets

def svc_interact(C, gamma, model):
    status_widget.value = 'Calculating...'
    
    model.named_steps.svc.set_params(**{'C': C, 'gamma': gamma})

    model.fit(X_train, y_train)
    plot_2DClassifier(X_train, selected_features, y_train, model, "SVM w/ training set")
    
    status_widget.value = f'Test set accuracy : {model.score(X_test, y_test)}'

C_widget = widgets.FloatLogSlider(value=10., min=-3, max=2, base=10, step=0.2, description='C:', disabled=False, continuous_update=False, orientation='horizontal', readout=True, readout_format='.3f')
gamma_widget = widgets.FloatLogSlider(value=0.25, min=-3, max=2, base=10, step=0.2, description='Gamma:', disabled=False, continuous_update=False, orientation='horizontal', readout=True, readout_format='.3f')
status_widget = widgets.Label(value='')

bi_svm = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

interact(svc_interact, C=C_widget, gamma=gamma_widget, model=fixed(bi_svm))
display(status_widget)

# %% [markdown]
# ## Feature uncertainty experiment
# - Assume above the decision boundary(model) as a model a data annotator uses to label trocical hurricanes
# - features X also as true measurements
# - Re-classify X with the model to have truely annotated labels (wrt. an annotator)
#
# (Re-classify the features with the model to have a perfectly separable dataset and then add Gaussian noise the the features to see the outcome.)

# %%
true_y = bi_svm.predict(X)

plot_2DClassifier(X, selected_features, true_y, bi_svm, "True measurements wrt. annotator")


# %% [markdown]
# ### Add Gaussian noise to features
# (Expected: Affecting data points in the vicinity of decision boundaries)

# %% editable=true slideshow={"slide_type": ""}
def noise_interact(sd):
    status_widget.value = 'Calculating...'
    noisy_X = X + np.random.normal(loc=0.0, scale=sd, size=X.shape)
    
    plot_2DClassifier(noisy_X, selected_features, true_y, bi_svm, "Noisy measurement wrt. annotator")

    noisy_pred = bi_svm.predict(noisy_X)
    status_widget.value = f"Noisy inputs accuracy: {bi_svm.score(noisy_X, true_y)}"

sd_widget = widgets.FloatSlider(value=1., min=0.1, max=5., step=0.1, description='SD :',
    disabled=False, continuous_update=False, orientation='horizontal', readout=True, readout_format='.1f'
)
status_widget = widgets.Label(value='')

interact(noise_interact, sd=sd_widget)
display(status_widget)
