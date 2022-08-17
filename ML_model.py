#machine learning libraries and functions 
from unittest import TestResult
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# pandas 
import pandas as pd

# saving model 
import pickle

## load the datset
iris_bunch = load_iris() ## its a bunch object works like a dictionary
iris_df = pd.DataFrame(iris_bunch['data'], columns=iris_bunch['feature_names'])

target = iris_bunch['target']

## split the train and the test
X_train, X_test, y_train, y_test = train_test_split(iris_df, target, random_state=1, test_size=0.2)


#Create the model
logistic = LogisticRegression(max_iter=1000)

#train it
logistic.fit(X_train, y_train)

## prediction 
#print(logistic.predict(X_test))

## evaluate
#print(logistic.score(X_test, y_test))


## save the model 
pkl_file = 'logistic_model.p'

with open(pkl_file, 'wb') as file:
    pickle.dump(logistic, file)

print(iris_df.head(3))