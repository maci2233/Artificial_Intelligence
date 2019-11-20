import pandas as pd
import pickle
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

if __name__ == '__main__':
    test = pd.read_csv("fifa_test.csv", encoding='utf8')
    y_true = test.pop('Position')
    x = test.drop(test.columns[0], axis=1)

    #classifier = "decision_tree"
    classifier = "bagging"

    if classifier == "decision_tree":
        files = [
            "decision_tree_fifa_MD6Cg.pickle",
            "decision_tree_fifa_MD7Cg.pickle",
            "decision_tree_fifa_MD7Ce.pickle"]
    elif classifier == "bagging":
        #x, _ = make_classification(n_samples=1500, n_features=5, random_state=10)
        #files = ['random_forest_fifa_MD4Cg.pickle']
        files = ['bagging_fifa_MS7354MF15.pickle']

    for file in files:
        with open(file, "rb") as f:
            clf = pickle.load(f)
        y_pred = clf.predict(x)
        acc = accuracy_score(y_true, y_pred)
        if classifier == "decision_tree":
            print(f"Test Accuracy max_depth={clf.max_depth}, criterion={clf.criterion}")
        elif classifier == "bagging":
            print(f"Test Accuracy max_samples={clf.max_samples}, max_features={clf.max_features}")
        print(acc)
        if file == "dec_tree_fifa_MD7Cg.pickle":
            dot_data = tree.export_graphviz(clf, out_file=None, feature_names=x.columns, filled=True, rounded=True)
            graph = graphviz.Source(dot_data)
            graph.render("mou")
