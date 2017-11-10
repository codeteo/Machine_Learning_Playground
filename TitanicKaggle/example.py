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
data_train, data_test = encode_features(data_train, data_test)

sns.barplot(x="Age", y="Survived", hue="Sex", data=data_train);

plt.show()
