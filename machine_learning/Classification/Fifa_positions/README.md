# Fifa 19 Player Classifier

### Introduction

Every player in Fifa has a total amount of 34 attributes that tell us how good they are at shooting, passing, defending, etc...
Based on these attributes, could it be possible to determine where in the field a player is expected to be?
We will try to answer this question by implementing two different classifiers that given any player's attributes, it will output its position.<br>
The dataset used for this project has more than 18 000 players' information and is available [here](https://www.kaggle.com/karangadiya/fifa19).<br>
The source codes for training and testing the models are available [here](https://github.com/maci2233/Artificial_Intelligence/tree/master/machine_learning/Classification/Fifa_positions).


### Data Pre-processing

In order to determine a player's position based on his attributes we only need the 2 things that were just mentioned (position, attributes).
The rest of the columns given in the data.csv are useless for us since they don't influence the output at all, therefore we are going to drop those columns.<br>
There's one more thing to do, the position column has the exact player's position, but for simplicity, we will only determine if the player
is an attacker, a midfielder, a defender, or a goalkeeper. For example, the positions LS, RS, ST, LW, RW, RF, LF and CF will be combined
in one single class that represents that attackers, and this process is repeated for the positions that represent midfielder, defender
and goalkeeper.<br>
At the end we will have the following classes represented with numerical values:
* Attacker - 0
* Midfielder - 1
* Defender - 2
* Goalkeeper - 3

### Traning and Test sets

The samples inside the data.csv will help us train our model, but the problem is that we have no players available for testing and checking how good we can classify a set of never seen players, to solve this, we will split the dataset in two different csv files

1. fifa_train.csv will have 90% of the players that will be used to train the model
2. fifa_test.csv will have the remaining 10% that will be used for testing

### First Classifier: Decision Tree

A decision tree is a very simple yet powerful classifier that allows us to make decisions based on different input values that it receives.
A tree contains several nodes, and each of them is in charge of taking a decision based on something; in our case, each node will
receive as an input a player's attribute and decide where to go next based on the value that the attribute has.

### Cross-Validation

Cross-validation is a very useful technique used to split the training dataset in two, so that one part trains
the model, and the other one validates it. This validation step is pretty much testing the model and
measuring the and accuracy in our case, but the very important difference is that this testing is done
without using the testing dataset. This process is repeated for any given folds, so at the end when we finish
validating, we will have tested our model several times against unseen data.

After we are done with cross-Validation, we will graph the training accuracy and validation accuracy of each fold, this will allow us to notice 2 different things:
1. If the model is unstable and behaves very different between multiple validation sets, the accuracy between folds will vary a lot, so we will visualy notice that
2. If the model is overfitted, that means that instead of fitting the training set, it just memorized it to have a much higher training accuracy, but since it does not generalize for other datasets, the validation accuracy will be much lower

Detecting these 2 factors is very important since based on the results we will decide which model to use and with which set of hyper-parameters. For example:

This is the result of the cross-valition using a decision tree with no max depth defined, in other words, a decision tree that uses all
34 player attributes to determine the class.

![Figure 1](https://github.com/maci2233/Artificial_Intelligence/blob/master/machine_learning/Classification/Fifa_positions/CV_no_max_depth.png)

As we can see, the training accuracy for each fold is pretty much 100%, but thanks to the cross-validation results we know that the model is overfitted, since the accuracy for the validation sets is around 81% in average. To solve this we are going to create several trees with different hyper-parameter values and compare their cross-validation results. The two hyper-parameters that will be modified are:
* max_depth: Limits the amount of features that the tree can use to determine a class
* criterion: There are some player attributes that are more important than others to know the position. The criterion is the strategy
that the tree will use to determine which attributes are the most important

There are a lot more things that can be modified but we will play around with only these 2 values to keep it simple.

### Selecting the best models

All the results are going to be stored and then we are going to choose the best models according to 2 different categories:
1. Accuracy: We want to know which models performed better on average against unseen data
2. Overfit Score: We will check this value to know which models have similar accuracies between the training and validation sets. In order to get this value for each model we just do this operation: ((sum of training_scores) - (sum of validation_scores)) / (number of folds) this way we will know on average how close the training accuracy is from the validation accuracy.

These are the results obtained:

![Figure 2](https://github.com/maci2233/Artificial_Intelligence/blob/master/machine_learning/Classification/Fifa_positions/CV_results.png)

As we can see, there are 3 Decision trees that appear in both categories:
* max_depth = 7, criterion = "gini"
* max_depth = 6, criterion = "gini"
* max_depth = 7, criterion = "entropy"

This is the result obtained doing cross-validation for one of these decision trees:

![Figure 3](https://github.com/maci2233/Artificial_Intelligence/blob/master/machine_learning/Classification/Fifa_positions/CV_MD7Cg.png)

We can see how the training and validation accuracies are way closer than the results obtained in the other Decision tree.<br>
The average accuracy for the validation set is also higher (around 84%) because the fact we are limiting the max_depth of the model allows it to generalize better and not just learn the whole training samples. 

### Saving the models

We ended up with 3 final models that will be used for testing, we are going to save them and then make the testing in a separate file.
The reason for doing this is simple, training takes time, so having to train the models each time we want to test them makes no sense at all since it is very time consuming

### Testing the decision trees

Testing is pretty straightforward. For each of the 3 decision trees that we chose at the end, we will make a prediction for each sample in the test set, then we will compare all the predicted classes with the real classes and get the test accuracy to see how well the models perform.

Test accuracy results:

![Figure 4](https://github.com/maci2233/Artificial_Intelligence/blob/master/machine_learning/Classification/Fifa_positions/decision_tree_tests.PNG)

As we can appreciate, basically we can achieve an 85% accuracy using a decision tree. Can this number go higher using a different approach?

### Second Classifier: Bagging Classifier

A bagging classifier is a type of ensemble method, its purpose is to train several classifiers, each of them with a smaller random portion of the dataset and then for each sample, the final prediction of its class is determined based on a voting system, so if there are 3 classifiers and 2 of them vote for class A while the other votes for class B, then the final predicion is A.

Ensemble methods can achieve higher accuracies than a single model, since they have several models training on different samples and different features of the training dataset, at the end there are a lot of models that may not be the best individually, but since each of them recognizes some simple patterns, once we do the voting for each sample, the chances of outperforming the decision tree that we just trained will be high.

### Repeating previous steps

Even though the classifier is different, the steps for training the model and choosing the best one using cross validation are pretty much the same, so we will skip that part and just compare the results.

### Testing the bagging classifier

Our final model has 250 decision trees that are trained using 45% of the training dataset and 45% of the total amount of features, that's why each tree is able to learn different patterns than others.

Test accuracy results:

![Figure 5](https://github.com/maci2233/Artificial_Intelligence/blob/master/machine_learning/Classification/Fifa_positions/bagging_tests.PNG)

We just achieved an 88% accuracy with the bagging classifier, which is higher than the individual decision tree. Depending on the problem, sometimes one approach is better than the other so it is a good idea to try several models (if time allows it) and not just stick to one and compare their results.

