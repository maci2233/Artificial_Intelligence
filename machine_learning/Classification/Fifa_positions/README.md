# Fifa 19 Player Classifier

### Introduction

Every player in Fifa has a total amount of 34 attributes that tell us how good they are at shooting, passing, defending, etc...
Based on these attributes, could it be possible to determine where in the field a player is expected to be?
We will try to answer this question by implementing two different classifiers that given any player's attributes, it will output its position.<br>
The dataset used for this project has more than 18 000 players' information and is available [here](https://www.kaggle.com/karangadiya/fifa19).<br>
The source codes for training and testing the models are available [here](https://github.com/maci2233/Artificial_Intelligence/tree/master/machine_learning/Classification/Fifa_positions).


### Data Pre-processing

In order to determine a player's position based on his attributes we only need the 2 things that were just mentioned (position, attributes).
The rest of the columns given in the data.csv is useless for us since it does not influence the output at all, therefore we are going to
drop those columns.<br>
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

As we can see, the training accuracy for each fold is pretty much 100%, but thanks to the cross-validation results we know that the model is overfitted, since the accuracy for the validation sets is around 82% in average. To solve this we are going to create several trees with different hyper-parameter values and compare their cross-validation results. The two hyper-parameters that will be modified
are the tree's maximum depth and its criterion. There are a lot more things that can be modified but we will play around with only these
two values to keep it simple.

All the results are going to be stored and then we are going to choose the best models according to 2 different categories:
1. Accuracy: We want to know which models performed better on average against unseen data
2: Overfit Score: We will check this value to know which models have similar accuracies between the training and validation sets. In order to get this value or each model we just do this operation: ((sum of training_scores) - (sum of validation_scores)) / (number of folds) this way we will know on average how close the training accuracy is from the validation accuracy

This are the results obtained:

PENDIENTE
