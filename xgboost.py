## Titanic competition in Kaggle
## V4 : XGBoost. Accuracy 0.77033

## Download data from Kaggle.
import pandas
from sklearn.model_selection import train_test_split
import numpy as np

train = pandas.read_csv('titanic_train.csv')
train.columns  
'''
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
'''  
X = train.drop('Survived',axis=1)
y = (train.copy())['Survived']
# Split the (training) data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4686)

# missing values in X_train
np.sum(X_train.isna(), axis=0)

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler

Impute1 = KNNImputer(n_neighbors=10, weights="distance")
Impute2 = SimpleImputer(strategy="most_frequent")
Encode = OneHotEncoder(handle_unknown="ignore")
Scale = StandardScaler()

Cat_nan = ['Embarked', 'Cabin']
Cat_attribs = ['Name','Sex','Ticket']
Num_attribs = ['Age', 'Fare']

num_pipiline1 = Pipeline(steps=[
    ('impute1',Impute1),
    ('scale1', StandardScaler())
])

num_pipiline2 = Pipeline(steps=[
    ('scale1', StandardScaler())
])

cat_pipeline1 = Pipeline(steps=[
    ('impute2', Impute2),
    ('enco', Encode)
])

cat_pipeline2 = Pipeline(steps=[ 
    ('encode', Encode)
])

total_pipeline = ColumnTransformer(transformers=[
    ('num_imp', num_pipiline1, Num_attribs),
     ('cat_imp', cat_pipeline1, Cat_nan),
    ('cat_encode', cat_pipeline2, Cat_attribs)
], remainder="passthrough")


import xgboost as xgb

xgb_prep = Pipeline(steps=[ 
    ('preproc', total_pipeline),
    ('xgb', xgb.XGBClassifier(tree_method = 'approx', enable_categorical = True))
])

# xgboost only support numerical and boolean data in general; with enable_categorical =True,
# it does have some support to categorical data for some tree methods.
# xgbm = xgb.XGBClassifier(n_estimators=5, n_jobs=-1, max_depth = 5,  enable_categorical = True)

xgb_prep.fit(X_train, y_train)
y_pred = xgb_prep.predict(X_train)
print(classification_report(y_train, y_pred))

xgb_cv = cross_val_score(xgb_prep, X_train, y_train, cv=5)
print(np.mean(xgb_cv))
# 0.80


xgbrf_prep = Pipeline(steps=[ 
    ('preproc', total_pipeline),
    ('xgb', xgb.XGBRFClassifier(tree_method = 'approx', learning_rate=0.5, n_estimators=10, max_depth=6, max_leafs=10, enable_categorical = True, random_state=4686))
])

xgbrf_prep.fit(X_train, y_train)
y_xgbrf_pred = xgbrf_prep.predict(X_train)
print(classification_report(y_train, y_xgbrf_pred))

'''
param_dist = {
    'xgb__tree_method': ['approx', 'hist', 'exact'],
    'xgb__learning_rate': np.arange(0.1, 4, 0.5),
    'xgb__n_estimators': np.arange(3, 20, 5),
    'xgb__max_depth': np.arange(5, 8),
    'xgb__max_leaves': [5, 10, 15, 20]
}

xgb_grid = GridSearchCV(xgb_prep, param_grid=param_dist, cv=10)
xgb_grid.fit(X_train, y_train)

print(classification_report(y_train, xgb_grid.predict(X_train)))
print(classification_report(y_test, xgb_grid.predict(X_test)))


#Further search. It takes quite long time; about one hour. 
param_dist2 = {
    'xgb__tree_method': ['approx'],
    'xgb__learning_rate': np.arange(0.1, 1.1, 0.05),
    'xgb__n_estimators': np.arange(5,15,1),
    'xgb__max_depth': [6],
    'xgb__max_leaves': np.arange(10, 100,5)
}
xgb_grid2 = GridSearchCV(xgb_prep, param_grid=param_dist2, cv=10)
xgb_grid2.fit(X_train, y_train)

print(classification_report(y_train, xgb_grid2.predict(X_train)))
#0.94
print(classification_report(y_test, xgb_grid2.predict(X_test)))
# 0.80
# The best_params_ are: approx, learning_rate 0.8, n_estimators 8, max_depth 6, and max_leaves 25.
'''

xgb2 = Pipeline(steps=[ 
    ('preproc', total_pipeline),
    ('xgb', xgb.XGBClassifier(tree_method = 'approx', learning_rate= 0.8, n_estimators=8, max_depth=6, max_leaves=25, enable_categorical = True))
])

xgb2.fit(X_train, y_train)
print(classification_report(y_train, xgb2.predict(X_train)))
print(classification_report(y_test, xgb2.predict(X_test)))

test = pandas.read_csv('titanic_test.csv')
# pred_test = random_forest.predict(test)
# Error: "Fare" has a missing value, in contrast to X_train and X_test.

Fare = test[['Fare']]

Fare_ny = SimpleImputer(strategy='mean').fit_transform(Fare)

test['Fare'] = Fare_ny

pred_test = xgb2.predict(test)

res1 = (test.copy())['PassengerId']
result = (pandas.DataFrame(res1)).assign(Survived = pred_test)
result.to_csv('submission.csv',index=False)
