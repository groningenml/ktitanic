# GroningenML
# Kaggle Titanic Competition


# Round 2: 22-05-2019
# Result: II 0,8268 (I 0,7978)
# ToDo Round 3:
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
df_test_after_modelling = raw_csv_data_test.copy() # PassengerId 892 - 1309
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
workbook = pd.ExcelWriter('output/workbook_xxx.xlsx', engine='xlsxwriter')
df_summary.to_excel(workbook, sheet_name='Summary numerical features', header=True, index=True)
df = df[sorted(df)] # sort to make it possible to work with the same index
df.to_excel(workbook, sheet_name='Raw data - sorted column names', header=True, index=True)
features.to_excel(workbook, sheet_name='Features', header=True, index=False)
workbook.save()
workbook.close()


# =============================================================================
# Part 2 - Dataset cleaning (features, observations)
# =============================================================================
## Drop non-relevant features
df = df.drop(columns = ['PassengerId', 'Name'])

## check op NaNs
df.isnull().values.any() # 
df.isna().any()
df.isnull().sum() 

## Age NaNs: take median
median = df['Age'].median() # 28.0
df['Age'].fillna(median, inplace=True)

## Cabin: add feature Cabin_bin yes (1) or no (0) and drop Cabin
df['Cabin'].fillna(0, inplace=True) # Change NaN to 0
df['Cabin_bin'] = df['Cabin']
for i in range (df.shape[0]):
    if df.Cabin_bin[i] != 0:
        df.Cabin_bin[i] = 1
df = df.drop(columns = ['Cabin'])
df['Cabin_bin'] = df['Cabin_bin'].astype(int)

## Embarked: add feature Embarked_bin yes (1) or no (0)
df['Embarked'].fillna(0, inplace=True) # Change NaN to 0
df['Embarked_bin'] = df['Embarked']
for i in range (df.shape[0]):
    if df.Embarked_bin[i] != 0:
        df.Embarked_bin[i] = 1
df['Embarked_bin'] = df['Embarked_bin'].astype(int)

## Take care of objects 
# Sex: LabelEncoder
from sklearn.preprocessing import LabelEncoder
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
# Embarked: One Hot Encoding (including avoiding dummy variables trap)
df = pd.get_dummies(df, columns=['Embarked'], prefix=["Embarked"], drop_first=True) 

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

## Balancing the Training Set
# create lists of y_train = 1 and y_train = 0
pos_indices = y_train[y_train.values == 1].index
neg_indices = y_train[y_train.values == 0].index
# determine higher (more observations) and lower (less) list
if len(pos_indices) > len(neg_indices):
    higher_list_indices = pos_indices
    lower_list_indices = neg_indices
else:
    higher_list_indices = neg_indices
    lower_list_indices = pos_indices  
# update the higher list with the length of the lower list no. of observations
import random
random.seed(0)
higher_list_indices = np.random.choice(higher_list_indices, size=len(lower_list_indices)) # subset
lower_list_indices = np.asarray(lower_list_indices)
new_indices = np.concatenate((lower_list_indices, higher_list_indices))
# update X_train and y_train (balanced on y values)
X_train = X_train.loc[new_indices,]
y_train = y_train[new_indices]
# check distrubution on y (should be equal now)
y_train.value_counts()
#1    273
#0    273

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
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred)) # Test Data Accuracy: I: 0.7697 / II: 0.8045
# Write to Model Selection
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
results = pd.DataFrame([['Logistic Regression', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])


# =============================================================================
# Part 3 - MODEL2: RandomForest
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
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred)) # Test Data Accuracy: I: 0.7978 / II: 0.8268
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
classifier = XGBClassifier(max_depth = 10, learning_rate = 0.3, n_estimators = 400)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

## Result
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# Heatmap version 1
sns.heatmap(cm,annot=True,fmt="d")
# Accuracy
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred)) # Test Data Accuracy: I: 0.7697 / II: 0.7765
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
# Write result to excel and csv
# =============================================================================
results = results.sort_values(['Accuracy'], ascending = False)
results.to_excel(r'output/results.xlsx', index = False)
results.to_csv(r'output/results.csv', index = False)

