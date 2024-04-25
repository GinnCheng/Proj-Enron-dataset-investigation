import sys
import pickle
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(("../tools/")))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

### define some functions if it needs to be called for multiple times
def print_uniq_val(df):
    for col in df.columns:
        print(col,':',df[col].unique())

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

## turn the dictionary into dataframe to have better look
pd.set_option('display.max_columns',100)
df = pd.DataFrame(data_dict).T
## now we could see that mostly that 'NaN' should be replaced or imputed
df.replace('NaN',np.nan, inplace=True)

# sort the number of null values and get the feature names
feature_low_null = df.isnull().sum().sort_values().keys()
# select the top 12 columns with the least null values
feature_low_null[:12]
# actively remove some columns that are not very informative
feature_low_null = [col for col in feature_low_null if col not in ['email_address','other']]
df = df[feature_low_null]

## select the float columns to apply imputation
num_col = [col for col in df.columns if df[col].dtype == float]
col_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
## data imputation
df[num_col] = col_imputer.fit_transform(df[num_col])
print(df.isnull().sum().sum())

## using model to select the most relavant features
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
df_features = df.copy()
df_features.drop(columns=['poi'], axis=1, inplace=True)
df_labels   = df.copy().poi
## get the KBest selector
feature_selector = SelectKBest(mutual_info_classif, k=6)
feature_selector.fit(df_features, df_labels)

## select the top 5 most relevant columns to poi as per f1 class
features_list = feature_selector.get_feature_names_out().tolist()
features_list = ['poi'] + features_list
features_list

my_dataset = df.T.to_dict()

## in this project I will still use the featureFormat, targetFeaturesSplit
## instead of directly using DataFrame just for project code consistency
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

## scale the features data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features,labels)
pred = clf.predict(features)
print('Naive bayes')
print(f'The precision is {precision_score(labels, pred)}')
print(f'The recall is {recall_score(labels, pred)}')

from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=10000.0, gamma='auto')
clf.fit(features, labels)
pred = clf.predict(features)
print('SVM classifier')
print(f'The precision is {precision_score(labels, pred)}')
print(f'The recall is {recall_score(labels, pred)}')

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_leaf=1)
clf.fit(features, labels)
pred = clf.predict(features)
print('Decision tree classifier')
print(f'The precision is {precision_score(labels, pred)}')
print(f'The recall is {recall_score(labels, pred)}')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=1)
clf.fit(features, labels)
pred = clf.predict(features)
print('Random Forest Classifier')
print(f'The precision is {precision_score(labels, pred)}')
print(f'The recall is {recall_score(labels, pred)}')

clf = AdaBoostClassifier(n_estimators=127, learning_rate=0.22)
clf.fit(features, labels)
pred = clf.predict(features)
print('AdaBoost Classifier')
print(f'The precision is {precision_score(labels, pred)}')
print(f'The recall is {recall_score(labels, pred)}')

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split, RepeatedKFold
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
del clf
clf = AdaBoostClassifier(n_estimators=10, learning_rate=0.11)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
print('To train: ')
print(f'The precision is {precision_score(labels_test, pred)}')
print(f'The recall is {recall_score(labels_test, pred)}')

# turn labels into arrray for better using boolean index
labels = np.array(labels)

# Repeated KFold method
rkf = RepeatedKFold(n_splits=2, n_repeats=100, random_state=42)
rkf_precision, rkf_recall = [], []
# split the index for training
for train_index, test_index in rkf.split(features):
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    # now fit the data
    clf = AdaBoostClassifier(n_estimators=20, learning_rate=0.21)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    tmp_precision, tmp_recall = precision_score(y_test, pred), recall_score(y_test, pred)
    rkf_precision.append(tmp_precision)
    rkf_recall.append(tmp_recall)


from matplotlib import pyplot as plt
plt.figure()
plt.scatter(rkf_precision, rkf_recall, color='blue')
plt.plot([0,1], [0.3,0.3], color='k', linestyle='--')
plt.plot([0.3,0.3], [0,1], color='k', linestyle='--')
ax = plt.gca()
ax.set_xlabel('Precision score')
ax.set_ylabel('Recall score')
ax.set_title('The distribution of precision and recall of all trainings')
plt.show()

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# make the pipeline
clf = Pipeline([
    ('preprocessor', MinMaxScaler()),
    ('classifier', AdaBoostClassifier(n_estimators=20, learning_rate=0.21))
])

# test the pipeline
clf.fit(features_train,labels_train)
clf.score(features_test,labels_test)
