import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')

print(data_train.sample(3))

def drop_features(df):
	return df.drop(['Ticket', 'Fare', 'Name', 'Embarked', 'Parch', 'SibSp', 'Cabin'], axis=1)

def simplify_ages(df):
	df.Age = df.Age.fillna(-0.5)
	bins = (-1, 0, 5, 18, 60, 100)
	group_names = ['Unknown', 'Baby', 'Teenager', 'Adult', 'Senior']
	categories = pd.cut(df.Age, bins, labels=group_names)
	df.Age = categories
	return df

def simplify_sex(df):
	df.Sex = df.Sex.fillna('NaN')
	return df

from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Age', 'Sex']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
data_train = drop_features(data_train)
data_train = simplify_ages(data_train)
data_train = simplify_sex(data_train)
data_test = drop_features(data_test)
data_test = simplify_ages(data_test)
data_test = simplify_sex(data_test)
data_train, data_test = encode_features(data_train, data_test)


# Time for some ML
from sklearn.model_selection import train_test_split

X_all = data_train.drop(['Survived'], axis=1)
y_all = data_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

sns.barplot(x="Age", y="Survived", hue="Sex", data=data_train);

# plt.show()

# fitting and tuning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# choose classifier
clf = RandomForestClassifier()

params = {'n_estimators':[4, 6, 9], 'max_features':['log2', 'sqrt', 'auto'],
			'criterion':['entropy', 'gini'], 'max_depth': [2, 3, 5, 10],
			'min_samples_split':[2, 3, 5], 'min_samples_leaf':[1, 5, 8]
		 }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, params, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# set the clf to the best combination of params
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))
