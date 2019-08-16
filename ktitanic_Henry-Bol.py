# GroningenML
# Kaggle Titanic Competition


# Round 3: 23-05-2019
# Result: III 0.8771 (II 0.8268 I 0.7978)
# Exclude Balancing: 
# Logistic Regression: 0.8212
# Random Forest: 0.8659
# XGBoost: 0.8771 (default hyperparameters)
#
# ToDo Round 4:
# Cabin: check non-numerical (categorical?)
# Name: (temp dropped)
# Ticket: check relation
# GridSearch (especially improve on XGBoost)
# Relations: Ticket, Name, SibSp, Parch, Cabin

# =============================================================================
# Part 1 - Data Preparation
# =============================================================================
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
raw_csv_data_train = pd.read_csv('Data/train.csv', sep = ',')
raw_csv_data_test = pd.read_csv('Data/test.csv', sep = ',')

# Copy to DataFrame ((test and train))
df_train = raw_csv_data_train.copy() # PassengerId 1 - 891
df_kaggle_test = raw_csv_data_test.copy() # PassengerId 892 - 1309
df = df_train

# Overview of dataset
df.head()
df.info() #
df.shape # (891, 12)
df_summary = df.describe()
features = pd.DataFrame(df.columns.values, columns = ['name'])
features['type'] = df.dtypes.values
features['rows #'] = df.count().values # non-NaNs rows
features['NaN #'] = df.isnull().sum().values
features['unique #'] = df.nunique().values
features = features.sort_values(['name'])

## Create workbook (and sort df on feature names alphabetically)
#workbook = pd.ExcelWriter('output/workbook_xxx.xlsx', engine='xlsxwriter')
#df_summary.to_excel(workbook, sheet_name='Summary numerical features', header=True, index=True)
#df = df[sorted(df)] # sort to make it possible to work with the same index
#df.to_excel(workbook, sheet_name='Raw data - sorted column names', header=True, index=True)
#features.to_excel(workbook, sheet_name='Features', header=True, index=False)
#workbook.save()
#workbook.close()


# =============================================================================
# Part 2 - Dataset cleaning (features, observations)
# =============================================================================
## Drop non-relevant features
df = df.drop(columns = ['PassengerId', 'Name'])
df_kaggle_test_PassengerId = df_kaggle_test.PassengerId
df_kaggle_test = df_kaggle_test.drop(columns = ['PassengerId', 'Name'])

## check op NaNs
df.isnull().values.any() # 
df.isna().any()
df.isnull().sum() 

df_kaggle_test.isnull().values.any() # 
df_kaggle_test.isna().any()
df_kaggle_test.isnull().sum() 


## Age NaNs: take median
median = df['Age'].median() # 28.0
df['Age'].fillna(median, inplace=True)
df_kaggle_test['Age'].fillna(median, inplace=True)

## Fare NaNs: take median
median = df['Fare'].median() # 28.0
df_kaggle_test['Fare'].fillna(median, inplace=True)


## Cabin: add feature Cabin_bin yes (1) or no (0) and drop Cabin
df['Cabin'].fillna(0, inplace=True) # Change NaN to 0
df['Cabin_bin'] = df['Cabin']
for i in range (df.shape[0]):
    if df.Cabin_bin[i] != 0:
        df.Cabin_bin[i] = 1
df = df.drop(columns = ['Cabin'])
df['Cabin_bin'] = df['Cabin_bin'].astype(int)

df_kaggle_test['Cabin'].fillna(0, inplace=True) # Change NaN to 0
df_kaggle_test['Cabin_bin'] = df_kaggle_test['Cabin']
for i in range (df_kaggle_test.shape[0]):
    if df_kaggle_test.Cabin_bin[i] != 0:
        df_kaggle_test.Cabin_bin[i] = 1
df_kaggle_test = df_kaggle_test.drop(columns = ['Cabin'])
df_kaggle_test['Cabin_bin'] = df_kaggle_test['Cabin_bin'].astype(int)


## Embarked: add feature Embarked_bin yes (1) or no (0)
df['Embarked'].fillna(0, inplace=True) # Change NaN to 0
df['Embarked_bin'] = df['Embarked']
for i in range (df.shape[0]):
    if df.Embarked_bin[i] != 0:
        df.Embarked_bin[i] = 1
df['Embarked_bin'] = df['Embarked_bin'].astype(int)

df_kaggle_test['Embarked'].fillna(0, inplace=True) # Change NaN to 0
df_kaggle_test['Embarked_bin'] = df_kaggle_test['Embarked']
for i in range (df_kaggle_test.shape[0]):
    if df_kaggle_test.Embarked_bin[i] != 0:
        df_kaggle_test.Embarked_bin[i] = 1
df_kaggle_test['Embarked_bin'] = df_kaggle_test['Embarked_bin'].astype(int)


## Take care of objects 
# Sex: LabelEncoder
from sklearn.preprocessing import LabelEncoder
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
# Embarked: One Hot Encoding (including avoiding dummy variables trap)
df = pd.get_dummies(df, columns=['Embarked'], prefix=["Embarked"], drop_first=True) 

df_kaggle_test['Sex'] = LabelEncoder().fit_transform(df_kaggle_test['Sex'])
# Embarked: One Hot Encoding (excluding avoiding dummy variables trap: Test set does not have NaNs)
#df_check = df_kaggle_test.copy()
#df_kaggle_test = df_check.copy()
df_kaggle_test = pd.get_dummies(df_kaggle_test, columns=['Embarked'], prefix=["Embarked"], drop_first=False) 


## Ticket: remove non-numerical characters
import re
i = 0
df_length = len(df)
for i in range (0, df_length):
    df.loc[i, 'Ticket'] = re.sub('[^0-9]','', df.loc[i, 'Ticket'])
# change from object to int
df['Ticket'] = pd.to_numeric(df['Ticket'], errors='coerce')
df.isnull().sum() 
# Ticket 4 -> 4 observations with Ticket NaN
#df = df.dropna(subset = ['Ticket'])
df['Ticket'].fillna(0, inplace=True) # Change NaN to 0

i = 0
df_kaggle_test_length = len(df_kaggle_test)
for i in range (0, df_kaggle_test_length):
    df_kaggle_test.loc[i, 'Ticket'] = re.sub('[^0-9]','', df_kaggle_test.loc[i, 'Ticket'])
# change from object to int
df_kaggle_test['Ticket'] = pd.to_numeric(df_kaggle_test['Ticket'], errors='coerce')
df_kaggle_test.isnull().sum() 
# Ticket 4 -> 4 observations with Ticket NaN
#df_kaggle_test = df_kaggle_test.dropna(subset = ['Ticket'])
df_kaggle_test['Ticket'].fillna(0, inplace=True) # Change NaN to 0


# =============================================================================
# Part 3 - EDA
# =============================================================================
## Check distribution in target feature
sns.countplot(df['Survived'], label = 'Count') 
plt.savefig('output/feature_target_distribution.jpg')
# > balance on Survived 

## Correlation with Response Variable (linear)
df2 = df.dropna(subset = ['Survived']) # check subset with Survival only (i.e Training set)
df2 = df2.copy().drop(columns = ['Survived']) # correlation with feature target
df2.corrwith(df.Survived).plot.bar(
            figsize=(20,10), # correlation with response variable / barplot
            title = 'Correlation with response variable',
            fontsize = 10, rot = 45,
            grid = True)
plt.savefig('output/data_correlation_response.jpg')

## Correlation Matrix between features (check independency between independent numerical features)
sns.set(style="white", font_scale=1)
# Compute the correlation matrix
corr = df2.corr()
# Generate a mask for the upper triangle (keep only lower end)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig('output/data_correlation_matrix.jpg')


# =============================================================================
# Part 4 - Preparation for modelling
# =============================================================================
X = df.drop(columns = ['Survived'])
y = df['Survived']
 
## Splitting the dataset into the Training set and Test set (this is not the test set for final validation)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = True)
## Splitting the Test dataset into the Train set and Validation set
#X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = 0.25, random_state = 0, shuffle = True)

### Balancing the Training Set (do not activate -> lower performance)
## create lists of y_train = 1 and y_train = 0
#pos_indices = y_train[y_train.values == 1].index
#neg_indices = y_train[y_train.values == 0].index
## determine higher (more observations) and lower (less) list
#if len(pos_indices) > len(neg_indices):
#    higher_list_indices = pos_indices
#    lower_list_indices = neg_indices
#else:
#    higher_list_indices = neg_indices
#    lower_list_indices = pos_indices  
## update the higher list with the length of the lower list no. of observations
#import random
#random.seed(0)
#higher_list_indices = np.random.choice(higher_list_indices, size=len(lower_list_indices)) # subset
#lower_list_indices = np.asarray(lower_list_indices)
#new_indices = np.concatenate((lower_list_indices, higher_list_indices))
## update X_train and y_train (balanced on y values)
#X_train = X_train.loc[new_indices,]
#y_train = y_train[new_indices]
## check distrubution on y (should be equal now)
#y_train.value_counts()
##1    273
##0    273

## Feature Scaling (while keeping the column names and row indices)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values # column names toevoegen
X_test2.columns = X_test.columns.values 
X_train2.index = X_train.index.values # juiste row indices toevoegen
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2

# Scaling on Kaggle Test set
X_kaggle_test = pd.DataFrame(sc_X.transform(df_kaggle_test))
X_kaggle_test.columns = X_train.columns.values # column names toevoegen

## importing the libraries
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# =============================================================================
# Part 3 - MODEL1: Logistic Regression
# =============================================================================
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, penalty = 'l1', solver = 'liblinear', multi_class = 'ovr')
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

## Result
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# Heatmap
sns.heatmap(cm,annot=True,fmt="d")
# Accuracy
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred)) # Test Data Accuracy: I: 0.7697 / II: 0.8045 / III: 0.8212
# Write to Model Selection
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
results = pd.DataFrame([['Logistic Regression', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])


# =============================================================================
# Part 3 - MODEL2: Random Forest
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100,
                                    random_state = 0,
                                    verbose = 4,
                                    n_jobs = -1)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

## Result
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# Heatmap
sns.heatmap(cm,annot=True,fmt="d")
# Accuracy
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred)) # Test Data Accuracy: I: 0.7978 / II: 0.8268 / III: 0.8659
# Write to Model Selection
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results = results.append(model_results, ignore_index = True, sort = False)


# =============================================================================
# Part 3 - MODEL3: XGBoost Classifier
# =============================================================================
from xgboost import XGBClassifier
#classifier = XGBClassifier(max_depth = 10, learning_rate = 0.3, n_estimators = 400)
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

## Result
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# Heatmap version 1
sns.heatmap(cm,annot=True,fmt="d")
# Accuracy
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred)) # Test Data Accuracy: I: 0.7697 / II: 0.7765 / III: 0.8771
# Determine features importance
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(classifier, height=0.8, ax=ax)
plt.show() # See XGBoost_feature_importance.png
plt.savefig('output/feature_importance_XGBoost.jpg')
# Write to Model Selection
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
model_results = pd.DataFrame([['XGBoost', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results = results.append(model_results, ignore_index = True, sort = False)


# =============================================================================
# Part 3 - MODEL4: CatBoost Classifier
# =============================================================================
from catboost import CatBoostClassifier
#classifier = XGBClassifier(max_depth = 10, learning_rate = 0.3, n_estimators = 400)
classifier = CatBoostClassifier()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

## Result
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# Heatmap version 1
sns.heatmap(cm,annot=True,fmt="d")
# Accuracy
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred)) # Test Data Accuracy: III: 0.8380 -> Not better
# Write to Model Selection
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
model_results = pd.DataFrame([['CatBoost', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results = results.append(model_results, ignore_index = True, sort = False)

# =============================================================================
# Part 4 - GridSearch
# =============================================================================
from sklearn.model_selection import GridSearchCV

#{'model__learning_rate': 0.05, 'model__max_depth': 3, 'model__n_estimators': 600, 'preprocessor__num__strategy': 'most_frequent'}
#parameters = {
##              'model__learning_rate': [0.005, 0.01, 0.02],
#              'model__learning_rate': [0.05],
##              'model__max_depth': [1, 2, 3, 4],
#              'model__max_depth': [3],
##              'model__n_estimators': [1, 2, 3, 4, 5, 10],
#              'model__n_estimators': [5],
#              'model__min_child_weight': [1, 5, 10],
#              'model__gamma': [0.5, 1, 1.5, 2, 5],
#              'model__subsample': [0.6, 0.8, 1.0],
#              'model__colsample_bytree': [0.6, 0.8, 1.0]
#}

parameters = {'min_child_weight': [1, 5, 10],
              'gamma': [0.5, 1, 1.5, 2, 5],
              'subsample': [0.6, 0.8, 1.0],
              'colsample_bytree': [0.6, 0.8, 1.0],
              'max_depth': [3, 4, 5]}

classifier = XGBClassifier()
gridsearch = GridSearchCV(estimator = classifier, 
                  param_grid = parameters, 
                  scoring = 'accuracy', 
                  verbose = 4,
                  cv = 10,
                  n_jobs= -1)

# Check gridssearch parameters
#gridsearch.get_params().keys()
#sorted(pipeline.get_params().keys())

import time
t0 = time.time()
gridsearch.fit(X_train, y_train)   
t1 = time.time() 
print("Took %0.2f seconds" % (t1 - t0)) #  
print('Best score and parameter combination = ')
print(gridsearch.cv_results_)    
print(gridsearch.best_score_)    
print(gridsearch.best_params_)    
# I XGBoost: {'model__learning_rate': 0.03, 'model__max_depth': 2, 'model__n_estimators': 50}
# II XGBoost: {'model__learning_rate': 0.01, 'model__max_depth': 2, 'model__n_estimators': 5}
# III XGBoost: {'model__learning_rate': 0.005, 'model__max_depth': 1, 'model__n_estimators': 1}
#=> no impacy on accuracy
# IV {'model__colsample_bytree': 0.6, 'model__gamma': 0.5, 'model__learning_rate': 0.05, 'model__max_depth': 3, 'model__min_child_weight': 1, 'model__n_estimators': 5, 'model__subsample': 0.6}
# V {'colsample_bytree': 0.6, 'gamma': 5, 'max_depth': 4, 'min_child_weight': 1, 'subsample': 0.8}
#=> worse result

# Predict
y_pred = gridsearch.predict(X_test)

# Evaluate the model 
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
model_results = pd.DataFrame([['GridSearch', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results = results.append(model_results, ignore_index = True, sort = False)


# =============================================================================
# Write result to excel and csv
# =============================================================================
#results = results.sort_values(['Accuracy'], ascending = False)
#results.to_excel(r'output/results220519a.xlsx', index = False)
#results.to_csv(r'output/results.csv', index = False)


# =============================================================================
# Step 6: Commit to Competion
# =============================================================================
# Preprocessing of test data, fit model
preds_test =  classifier.predict(X_kaggle_test)
preds_test =  gridsearch.predict(X_kaggle_test)


# Save test predictions to file
output = pd.DataFrame({'PassengerId': df_kaggle_test_PassengerId,
                       'Survived': preds_test})
output.to_csv('output/submission.csv', index=False)

