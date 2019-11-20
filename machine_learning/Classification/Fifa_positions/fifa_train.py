import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
#from sklearn.datasets import make_multilabel_classification
#from sklearn.preprocessing import MinMaxScaler

def combine_positions(row):
    #pos total = 26
    if row['Position'] in ['LS', 'RS', 'ST', 'LW', 'RW', 'RF', 'LF', 'CF']: #8
        return 0 #ATTACKER
    elif row['Position'] in ['LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LDM', 'CDM', 'RDM']: #11
        return 1 #MIDFIELDER
    elif row['Position'] in ['LWB', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']: #7
        return 2 #DEFENDER
    else:
        return 3 #GOALKEEPER


def preprocess_dataset(df):
    df.drop(axis=1, columns=['Unnamed: 0', 'Name', 'ID', 'Age', 'Photo', 'Nationality', 'Flag', 'Overall', 'Potential',
                             'Club', 'Club Logo', 'Value', 'Wage', 'Special', 'Preferred Foot', 'International Reputation',
                             'Weak Foot', 'Skill Moves', 'Work Rate', 'Body Type', 'Real Face', 'Jersey Number', 'Joined',
                             'Loaned From', 'Contract Valid Until', 'Height', 'Weight', 'LS', 'ST', 'RS', 'LW', 'LWB', 'LF', 'CF',
                             'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LDM', 'CDM', 'RDM',
                             'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Release Clause'], inplace=True)
    df['Position'] = df.apply(combine_positions, axis=1)
    #y = df.pop('Position')
    #scaler = MinMaxScaler()
    #df[df.columns] = scaler.fit_transform(df[df.columns])
    #df['Position'] = y
    df.dropna(inplace=True)
    return df

def cross_validation(clf, x, y, classifier, folds=10, plot_results=False):
    train_scores = []
    val_scores = []
    acc_difference = []
    fold_list = [f+1 for f in range(folds)]
    random_state = 10
    for fold in fold_list:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=random_state)
        clf.fit(x_train, y_train)
        train_score = clf.score(x_train, y_train)
        val_score = clf.score(x_val, y_val)
        train_scores.append(train_score)
        val_scores.append(val_score)
        acc_difference.append(train_score - val_score)
        random_state += 10
    if plot_results:
        plt.plot(fold_list, train_scores, label='train score', linewidth=3, color='red')
        plt.plot(fold_list, val_scores, label='validation score', linewidth=3, color='blue')
        if classifier == "decision_tree":
            plt.title(f'Cross validation results (max_depth={clf.max_depth}, {clf.criterion})')
        elif classifier == "bagging":
            plt.title(f'Cross validation results (max_samples={clf.max_samples}, max_features={clf.max_features})')
        plt.xlabel('Fold #')
        plt.ylabel('Score')
        plt.legend()
        plt.show()
    res = {'avg_train_acc': sum(train_scores)/len(train_scores),
           'avg_val_acc': sum(val_scores)/len(val_scores),
           'overfit_score': (sum(train_scores)-sum(val_scores))/len(train_scores)}
    if classifier == "decision_tree":
        res['max_depth'] = clf.max_depth
        res['criterion'] = clf.criterion
    elif classifier == "bagging":
        res['max_samples'] = clf.max_samples
        res['max_features'] = clf.max_features
    return res


def get_best_models(test, models_results, classifier, winners=5):
    best = []
    if test == 'overfit':
        if classifier == "decision_tree":
            overfit_scores = sorted([(model_result['overfit_score'], f"max_depth = {model_result['max_depth']}", f"criterion = {model_result['criterion']}") for model_result in models_results])
        elif classifier == "bagging":
            overfit_scores = sorted([(model_result['overfit_score'], f"max_samples = {model_result['max_samples']}", f"max_features = {model_result['max_features']}") for model_result in models_results])
        return overfit_scores[:winners]
    elif test == 'accuracy':
        if classifier == "decision_tree":
            val_accs = sorted([(model_result['avg_val_acc'], f"max_depth = {model_result['max_depth']}", f"criterion = {model_result['criterion']}") for model_result in models_results], reverse=True)
        elif classifier == "bagging":
            val_accs = sorted([(model_result['avg_val_acc'], f"max_samples = {model_result['max_samples']}", f"max_features = {model_result['max_features']}") for model_result in models_results], reverse=True)
        return val_accs[:winners]
    return best


if __name__ == '__main__':

    create_datasets = False #TRUE IF WE WANT TO GENERATE A TEST DATASET AND TRAIN DATASET

    if create_datasets:
        df = pd.read_csv("data.csv", encoding='utf8')
        df = preprocess_dataset(df)
        train, test = train_test_split(df, test_size=0.1, random_state=25)
        #print(train.columns)
        train.to_csv("fifa_train.csv")
        test.to_csv("fifa_test.csv")
    else: #DATASETS ARE ALREADY CREATED
        train = pd.read_csv("fifa_train.csv")
    y = train.pop('Position')
    if create_datasets:
        x = train
    else:
        x = train.drop(train.columns[0], axis=1) #The column Unnamed: 0 appears again so we drop it

    #classifier = "decision_tree"
    classifier = "bagging"

    validate = False #TRUE IF WE WANT TO DO CROSS VALIDATION OF THE MODEL, FALSE IF WE WANT TO SAVE IT

    if validate:
        clfs = []
        if classifier == "decision_tree":
            #We are going to create several trees to utilize different hyper-parameter values (max_depth, criterion) in this case
            max_depth = 3
            for i in range(10):
                clf1 = tree.DecisionTreeClassifier(max_depth=max_depth, criterion="gini")
                clf2 = tree.DecisionTreeClassifier(max_depth=max_depth, criterion="entropy")
                clfs.append(clf1)
                clfs.append(clf2)
                max_depth += 1
        elif classifier == "bagging":
            #samples, features = x.shape
            #x, y = make_multilabel_classification(n_samples=samples, n_features=features, n_classes=4, allow_unlabeled=False, random_state=10)
            #clfs = [RandomForestClassifier(n_estimators=500, max_depth=5, criterion="gini")]
            perc = 0.45
            clfs = [BaggingClassifier(tree.DecisionTreeClassifier(), n_estimators=250, max_samples=int(len(x.index) * perc), max_features=int(x.shape[1]*perc))]

        clfs_results = []
        folds = 10 #K FOLDS WANTED FOR CROSS VALIDATION
        for clf in clfs: #FOR EACH CLASSIFIER IN OUR CLASSIFIER LIST
            clf_results = cross_validation(clf, x, y, classifier, folds, plot_results=True)
            clfs_results.append(clf_results)
        best_accuracies = get_best_models(test='accuracy', models_results=clfs_results, classifier=classifier, winners=10)
        print("\n-----HIGHEST ACCURACY MODELS-----")
        for m in best_accuracies:
            print(m)
        less_overffited = get_best_models(test='overfit', models_results=clfs_results, classifier=classifier, winners=10)
        print("\n-----LESS OVERFITTED MODELS-----")
        for m in less_overffited:
            print(m)
    else:
        best_clfs = []
        if classifier == "decision_tree":
            #best_clfs.append(tree.DecisionTreeClassifier(max_depth=8, criterion="gini"))
            best_clfs.append(tree.DecisionTreeClassifier(max_depth=7, criterion="gini"))
            best_clfs.append(tree.DecisionTreeClassifier(max_depth=6, criterion="gini"))
            best_clfs.append(tree.DecisionTreeClassifier(max_depth=7, criterion="entropy"))
        elif classifier == "bagging":
            #x, y = make_classification(n_samples=1500, n_features=34, random_state=10)
            #best_clfs.append(RandomForestClassifier(n_estimators=1000, max_depth=4, criterion="gini"))
            perc = 0.45
            best_clfs.append(BaggingClassifier(tree.DecisionTreeClassifier(), n_estimators=250, max_samples=int(len(x.index) * perc), max_features=int(x.shape[1]*perc)))
        for clf in best_clfs:
            clf.fit(x, y)
            if classifier == "decision_tree":
                with open(f"{classifier}_fifa_MD{clf.max_depth}C{clf.criterion[0]}.pickle", "wb") as f:
                    pickle.dump(clf, f)
            elif classifier == "bagging":
                with open(f"{classifier}_fifa_MS{clf.max_samples}MF{clf.max_features}.pickle", "wb") as f:
                    pickle.dump(clf, f)
