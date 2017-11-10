import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')

print(data_train.sample(3))

sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train);

plt.show()
