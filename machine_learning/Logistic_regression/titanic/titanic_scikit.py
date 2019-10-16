import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import cross_validate


def prepare_dataset(filename, validate=True, train_filename='train.csv'):
    df = pd.read_csv(filename)
    #SURVIVED, CLASS, SEX, AGE, SIBSP, PARCH
    if validate:
        df.drop(axis=1, columns=['PassengerId', 'Name', 'Ticket', 'Fare', 'Embarked', 'Cabin'], inplace=True)
        mean_age = df['Age'].mean()
    else:
        df.drop(axis=1, columns=['Name', 'Ticket', 'Fare', 'Embarked', 'Cabin'], inplace=True)
        df_train = pd.read_csv(train_filename)
        mean_age = df_train['Age'].mean()

    df['Age'].fillna(mean_age, inplace=True)
    #df.dropna(axis=0, inplace=True)
    #scaler = MinMaxScaler()
    #df[['Age']] = scaler.fit_transform(df[['Age']])
    df = pd.get_dummies(df, columns=['Sex'])
    #df = pd.get_dummies(df, columns=['Pclass'])
    #SURVIVED, CLASS, AGE, SIBSP, PARCH, SEX_FEMALE, SEX_MALE
    return df


def cross_validation(lr, x, y, folds=5, plot_results=True):
    train_score = []
    test_score = []
    for i in range(folds):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        lr.fit(x_train, y_train)
        train_score.append(lr.score(x_train, y_train))
        test_score.append(lr.score(x_test, y_test))
    #cv_scores = cross_validate(lr, x, y, cv=folds, return_train_score=True)
    if plot_results:
        x_axis = np.array([i for i in range(1, folds+1)])
        plt.plot(x_axis, train_score, label='train score', linewidth=3, color='red')
        plt.plot(x_axis, test_score, label='test score', linewidth=3, color='blue')
        #plt.plot(x_axis, cv_scores['train_score'], label='train score', linewidth=3, color='red')
        #plt.plot(x_axis, cv_scores['test_score'], label='test score', linewidth=3, color='blue')
        plt.title('Cross validation results')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.legend()
        plt.show()
    else:
        return cv_scores


def logistic_regression(lr, x_train, y_train, x_test):
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    return y_pred

def generate_csv(ids, predictions):
    with open('titanic_sci_results.csv', 'w') as f:
        print('PassengerId,Survived', file=f)
        for id, pred in zip(ids, predictions):
            print('{},{}'.format(id, pred), file=f)

dataset = prepare_dataset('train.csv')
x = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]

lr = LogisticRegression(solver='lbfgs', penalty='l2', C=0.1, max_iter=500)

validate = True
if validate:
    #x_train, x_test, y_train, y_test = split_dataset(x, y)
    cross_validation(lr, x, y, folds=8)
else:
    test_dataset = prepare_dataset('test.csv', validate)
    pass_id = test_dataset['PassengerId']
    x_test = test_dataset.iloc[:, 1:]
    predictions = logistic_regression(lr, x, y, x_test)
    generate_csv(pass_id, predictions)
