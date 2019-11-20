# Fifa 19 Player Classifier

### Introduction

Every player in Fifa has a total amount of 34 attributes that tell us how good they are at shooting, passing, defending, etc...
Based on these attributes, could it be possible to determine where in the field a player is expected to be?
We will try to answer this question by implementing two different classifiers that given any player's attributes, it will output its position.<br>
The dataset used for this project has more than 18 000 players' information and is available [here](https://www.kaggle.com/karangadiya/fifa19).<br>
The source codes for training and testing the models are availabe [here](https://github.com/maci2233/Artificial_Intelligence/tree/master/machine_learning/Classification/Fifa_positions).


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

### First Classifier: Decision Tree

A decision tree is a very simple yet powerful classifier that allows us to make decisions based on different input values that it receives.
A tree contains several nodes, and each of them is in charge of taking a decision based on something; in our case, each node will
receive as an input a player's attribute and decide where to go next based on the value that the attribute has.
