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

# %% [markdown] id="dd193994-d69d-438f-a874-061e1dd4a01e"
# # North Atlantic Hurricanes
# The learning objectives of this notebook are to carry out feature selection for a classification model, train a classification model for types of hurricanes, and understand outcomes of feature uncertainty based on a experiment.
#
# The [North Atlantic Hurricanes dataset](https://myweb.fsu.edu/jelsner/temp/Data.html) used is developed by James B. Elsner and colleagues containing hurricanes recorded during the years 1944 to 2000. Each hurricane instance contains the year, the name (if it was named), the coordinates where it started, the last coordinates measured, the maximum coordinates (based on an aspect that an increment means closer to the coast), the maximum intensity, and the type. They are labeled into 3 types: tropical hurricanes (Type 0), hurricanes under baroclinic influences (Type 1), and hurricanes from baroclinic initiation (Type 3).
#
# Let's start with loading the data.

# %% id="46129f35-875c-4218-92d6-e26e91fcdc91" outputId="9aa2bf16-1fab-43bd-b9c6-5f08fb09485d"
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

# %% [markdown] id="9fd1fa0d-da1a-43cf-aec5-fa0676b4eeef"
# ## Introductory Data Inspection

# %% id="ac6fc382-8ee7-4fa8-86ce-d91aa3fe1dc4" outputId="fc2fd39e-e3ad-4742-bd94-97b98f4a2188"
hurricanes.info()

# %% [markdown] id="6f4c0458-1bd4-41a3-97f9-2f93ba15ef53"
# We will leave out the year and go on with first, last, and maximum cooridinates and maximum intensity as our potential fields for classification task.

# %% id="d17e94b0-59fc-44f7-908a-7b2912327900"
features = ['FirstLat','FirstLon','MaxLat','MaxLon','LastLat','LastLon','MaxInt']
label = ['Type']

# %% [markdown] id="e81c41ba-1d8d-4037-8376-0208a0c56da7"
# We can look at simple statistics of our features with `describe` function of pandas' dataframe.

# %% id="54c42903-67b9-4bd7-a6a4-a3409cf23d01" outputId="4c32c119-1e58-4175-e0fd-0e2721dfae85"
hurricanes[features].describe()

# %% [markdown] id="1e685207-c573-47d3-8f59-7668eae08119"
# Simply replacing `describe` with `boxplot` function, we can visualize some of stats above with box plot.

# %% id="a7bed402-9ac5-4dc3-9219-9f6b008cfc22" outputId="99e8b68d-7fea-4fca-e983-f8361f2ee0f0"
hurricanes[features].boxplot()
plt.show()

# %% [markdown] id="a0694f97-7cef-4676-b5d1-c4101132228c"
# ## Feature Selection
# In learning and working on machine learning or data science models in general, the size of the dataset can vary greatly. Large datasets refer to those containing a large number of data instances, a large number of features, or both. However, datasets containing a large number of features, also called a high dimensional datasets, present unique challenges because the larger they are, the bigger the feature space and the longer it takes algorithms to find the optimal solution to predict target variables. The inclusion of irrelevant and unimportant features can also negatively impact the performance of the model.
#
# Although the hurricane dataset we are working with is not very large, we can take this chance to apply feature selection methods to find the two most optimal features that will go into our classification model. Doing so will make it more convenient to visualize figures for a predictor on 2D plots. The following sections demonstrate two feature selection techniques including univariate selection and Random Forrest's feature importances.

# %% [markdown] id="aec0f434-a1e8-4fc9-a58b-63b6d9994d4d"
# ### Univariate Selection
#
# Statistical tests can be used to check how well each feature discriminates between classes of a target (categorical) variable. To quantify such tests, we use scikit-learn's ANOVA (analysis of variance) F-value `f_classif` together with `SelectKBest` to select features according to the k highest scores, in our case 2.

# %% id="5cdf4b7c-d8eb-4caa-adcf-d5bb1edead15" outputId="d81fac5b-0c86-4b30-ada2-a4a0dfa87966"
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

# %% [markdown] id="048515c6-947e-418e-9910-b22936c9f164"
# #### Intuition
# To understand how these statistical tests quantify features' discriminating scores, we plot the distributions of each hurricane type toward the feature with highest score `FirstLat` and the least one `FirstLon`.
#
# The graph shows that, even though the projected distributions onto `FirstLat` do not clearly discriminate between the 3 classes, they do much better than those of `FirstLon` where all class distributions are completely overlapped onto each other.

# %% id="48e70508-bdbe-4a0e-8e1d-719b6f9942b2" outputId="5024629d-66b8-429a-99e2-d1ffa2a5f456"
import seaborn as sns

sns.jointplot(data=hurricanes, x="FirstLon", y="FirstLat", hue="Type", palette="tab10")
plt.show()

# %% [markdown] id="91ff5e17-6e32-4791-b57b-2b0b3c3249dc"
# ### Random Forest's Feature Importance
# A Random Forest can be used to estimate the importances of features on a particular task. The importance of a feature is measured by looking the mean decrease in impurity, or the degree to which the tree nodes using that feature are able to reduce impurity (Gini or Shannon information gain) across all trees in the forest on average.

# %% id="f5c8ec49-05d7-44ed-9ba3-041af5a03435" outputId="f174a4ab-dd87-43ab-8f8c-d118a41ce0d2"
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

# %% [markdown] id="6192a4fd-fdf8-45ae-90dd-1b14ae5a6b8c"
# These results reflect an agreement between the two feature selection techniques, in that both methods selected first latitude, maximum latitude, and maximum intensity (`FirstLat`, `MaxLat`, and `MaxInt`) as the most optimal features for predicting the hurricane type.

# %% id="e06ef2d4-cc4a-4220-997f-6ac2f235201c"
selected_features = ['FirstLat', 'MaxLat', 'MaxInt']

# %% [markdown] id="77198df1-96b2-4680-bc6b-86a8ead9d86f"
# However, our goal is to have two features. The intuitive choice might be to simply pick the two highest scored features: `FirstLat` and `MaxLat`, but this strategy might result in the selection of redundant features.
#
# Feature Correlation is a useful approach that can be taken to further eliminate the less relevant features.

# %% [markdown] id="55118c9c-fad4-4f96-8c51-f2dd188fe3dc"
# ## Feature Correlation
# When two features are highly correlated, they are considered to provide the same knowledge about the target variables or labels, and therefore it is redundant to include both in a model.
#
# `FirstLat` and `MaxLat` are highly correlated. The data instances projected into both feature spaces also distribute toward our target variable `Type` very similarly.
#
# To avoid this redundancy, we will instead build the classification model with the features `FirstLat` and `MaxInt`.

# %% id="e319a821-ecac-4376-b05f-85f91c1a3c28" outputId="94a38b7d-25fc-4eef-f714-e4b9405204c7"
import seaborn as sns

fig, axs = plt.subplots(1,3, figsize=(20, 5))

# the abs. correlation matrix
df = pd.DataFrame(hurricanes, columns=selected_features)
abs_corr = df.corr().abs()
sns.heatmap(abs_corr, center=0, square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=axs[0])
axs[0].set_title("Absolute Correlations")

hurricanes.plot.scatter('FirstLat', 'Type', c='Type', colormap='coolwarm_r', ax=axs[1])
axs[1].set_title("FirstLat, Type")

hurricanes.plot.scatter('MaxLat', 'Type', c='Type', colormap='coolwarm_r', ax=axs[2], sharey=axs[1])
axs[2].set_title("MaxLat, Type")

plt.show()

# %% id="8be9dffc-a159-4310-abe8-957f796663bc"
selected_features = ['FirstLat', 'MaxInt']

# %% [markdown] id="de24b0e1-9513-4abc-b85c-6d4a1bddc003"
# ## Stratified Data Splitting

# %% [markdown] id="S3M0IDxvhK5s"
# When we plot the training samples according to Type, we see that there are a disproportionate samples for hurricanes of Type 0, with fewer hurricanes representing Type 1 and Type 3.

# %% id="19ba30cc-a569-4a97-93f2-54f8e5cf4bd2" outputId="7268b3c4-f25c-461d-eb35-91d8caa8b653"
hurricanes.groupby('Type').Type.count().plot(kind='bar', color='skyblue')
plt.show()

# %% [markdown] id="cYJK1i41ht2P"
# This could present a problem during training, because if we randomly sample the entire dataset, there is a chance that we will overselect Type 0 while failing to select enough Type 1 and Type 3. Such imbalances in the training samples could result in a model that has high prediction accuracy for Type 0 hurricanes, and low accuracy for the other types.
#
# To combat this issue, we can introduce stratification. Rather than randomly sampling from the entire dataset, stratification instead randomly samples from each strata (in this case, each Type) to achieve an adequate representation.

# %% id="6db1ef9b-915c-4774-9ea7-03c2a736bc6b"
from sklearn.model_selection import train_test_split

X,y = hurricanes[selected_features].copy(), hurricanes[label[0]].copy()
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=.7, random_state=0, stratify=y)

# %% [markdown] id="366dbfb3-19ea-41be-b994-295d98a8e5c7"
# ## Classification Model

# %% [markdown] id="UMxiyuaGa2r2"
# Support Vector Machines (SVMs) determine the hyperplane(s) that best separates the dataset into different categories. This is done, for example, by finding the hyperplane defined by the largest possible margin between data points in different categories.
#
# Here, the SVM is classified using C-Support Vector Classification (SVC), in order to handle multi-class classification. The “C” in the name refers to the penalty parameter of the error term.
#
# The features are also preprocessed to a 0 mean and unit variance by using the Standard Scaler. This is an important step because SVMs are not scale invariant.
#
# Two arrays are required for the input to the SVM: an array of training samples and an array of class labels.

# %% id="82522b7b-4496-4f9c-aad9-2f04ba5d1dba" outputId="dc8f7c70-6841-402f-87af-dff8a41f6e29"
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

svm = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

svm.fit(X_train, y_train)
svm.score(X_train, y_train)

# %% [markdown] id="YkkpcWY8bzw-"
# The SVM score indicates the mean accuracy of the predictions.

# %% id="61ce5bad-873d-4ae6-94ac-7c95f1f32155" outputId="870e9bc1-f3cd-43bd-f80c-7e15fc71eede"
svm.score(X_test, y_test)

# %% [markdown] id="krB_m2Rab1iT"
# In addition to seeing the score, we can also visualize the accuracy using a Confusion Matrix. It contains information about True Positives, True Negatives, False Positives and False Negatives, which in turn enable us to compute the precision and recall of the model.
#
# We can see from the Confusion Matrix that Type 0 hurricanes are estimated with the highest accuracy, while Type 1 hurricanes were often predicted as Type 3. This indicates that the hyperplane between Type 0 and the other types was better learned than the hyperplane between Type 1 and 3.

# %% id="c3767823-c503-4722-940c-ddc710c1edc5" outputId="ff8a3f99-4526-4dd1-93fe-a7dbbbccf9bd"
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

y_pred = svm.predict(X_test)

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation = 'vertical',
                                               cmap=plt.cm.Blues)
disp.figure_.suptitle("Confusion Matrix")

plt.show()

# %% [markdown] id="c0ff1689-f072-47fb-84c9-7b448c0c093c"
# ### Transform to binary classification
# As seen in the previous steps, the model predicts Type 0 tropical hurricanes well, but has difficulty distinguishing between Type 1 hurricanes under baroclinic influences, and Type 3 hurricanes from baroclinic initiation.
#
# Thus, we will revise the model from a multi-class classification to a binary classification that distinguishes between tropical and non-tropical hurricanes.

# %% id="83a09a5d-8bfa-4d83-a7c2-b80fff399326" outputId="18278ad1-de26-4b2e-9320-07cfe5d95d52"
tropical = hurricanes.copy()

# merge type 3 into type 1 for binary classification problem (tropical hurricane or not?)
tropical.loc[tropical['Type'] == 3, 'Type'] = 1
# check for binary type
tropical['Type'].unique()

# %% [markdown] id="QORatqrvAL-o"
# The Type 1 and Type 3 data samples are merged into a single type, and we perform a new train test split of the binary dataset.

# %% id="6f7a9b00-5372-4638-959d-aabcdc4d808d"
X,y = tropical[selected_features].copy(), tropical[label[0]].copy()
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=.7, random_state=0, stratify=y)

# %% id="70ed802b-28a9-45cd-9cdf-45c350d3591c"
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
    ax.legend(handles=handles, title='Type')
    plt.show()


# %% [markdown] editable=true id="cbe44cb1-d702-4213-b327-d7d92dcc64f1"
# ### SVM for binary classification

# %% [markdown] id="FHVD03pSAQPc"
# The hyperparameters of SVM (C and gamma) can be tuned using this widget to visualize how they effect the model results.
#
# C is a regularization parameter that determines the smoothness of the boundary between the categories. A larger C results in a larger proportion of correctly classified training samples, but the decision surface is less smooth and the margin is smaller. A smaller C results in a smoother decision boundary and larger margin at the expense of some misclassified samples.
#
# The gamma value determines the radius of influence of a single training sample. A small gamma value gives a single training point a wider radius of influence, with faraway points exerting higher influence. A large gamma value require that points be relatively close together to be included in the same category, with nearer points exerting a higher influence.
#
# Generally speaking, excessively high values result in overfitting, and excessively low values result in underfitting.

# %% editable=true colab={"referenced_widgets": ["aad2b8cc4bf642e5bc096f7b76ea054f", "0b14449dde02424098329bd430333529"]} id="3dafc668-c56e-47ab-aefe-72e77f81353b" outputId="e728982e-7a59-4121-9249-ad27af1f8d71"
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

# %% [markdown] id="a9338059-b680-436f-85a0-a0896ea31e26"
# ## Feature uncertainty experiment
# - Assume above the decision boundary(model) as a model a data annotator uses to label trocical hurricanes
# - features X also as true measurements
# - Re-classify X with the model to have truely annotated labels (wrt. an annotator)

# %% id="657cdf47-ba73-48bf-8f4a-484c3a566344" outputId="7842f721-16c0-4165-f0db-6ea6b16e01f0"
true_y = bi_svm.predict(X)

plot_2DClassifier(X, selected_features, true_y, bi_svm, "True measurements wrt. annotator")


# %% [markdown] id="e712651f-0967-4c58-85fc-11f141e15f9b"
# ### Add Gaussian noise to features
# - Expected: Affecting data points in the vicinity of decision boundaries to be the mislabeled data instances

# %% editable=true colab={"referenced_widgets": ["b46a465ed5a54a48ac27bb1255f6dd11", "fdbc6259cfd74a46bbe1bfedc70c6ee8"]} id="5f722163-ddff-4c01-bd19-0b7e63fe1f6a" outputId="46c9538c-1296-4122-d3c0-3d2ba515c1f2"
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
