# Classifying citation context based on purpose and influence

This repository contains the code used as a part of our team's submission for the 2021 3c Shared task. Task 1 was titled 'Citation Context Classification based on Purpose' where the Team (IREL) ranked first among the 22 participants on the leaderboard and Task 2 was aimed at  classifying citation context based on Influence where the team ranked second on the private leaderboard.


The repository contains the code for both the subtasks including all the experiments and their results on validation. 

## Experiment 1
###  Finetuning bert, roberta and scibert (both cased and uncased) with linear layer

For the first task we use a weighted loss function for this experiment. 
The training code can be run by `python3 first.py <model name> <batch size> <lr> <drop out> <file prefix>`


Example : `python3 first.py allenai/scibert_scivocab_uncased 4 0.00001 0 run1`

## Experiment 2 
### Running task1 with an unweighted loss function 

This experiment is only applicable to task 1 where we compare the results achieved by using weighted and unweighted loss functions.

The training code can be run by `python3 unweighted.py <model name> <batch size> <lr> <drop out> <file prefix>`

## Experiment 3
### finetuning scibert with LSTM for classification

Adding an LSTM layer after scibert instead of linear neural net layer. 

The training code can be run by `python3 third.py <model name> <batch size> <lr> <drop out> <file prefix>`

## Experiment 4
### Using Citing title with citation context for finetuning scibert 

Here we concatenate the citing title as well along with citation context and use it with an architecture similar to that of first experiment (scibert with a linear layer)


The training code can be run by `python3 fourth.py <model name> <batch size> <lr> <drop out> <file prefix>`


## Experiment 5
### Using Random forest for classification

We try to use random forest method to classify the embeddings reieved from scibert. 

The two hyperparameters involved are maximum tree depth and the number of trees in the forest which have been set to 35 and 1000 in the code provided

The training code can be run by `python3 fifth.py <file prefix>`


### Authors
[@him-mah10](https://github.com/him-mah10) and [@bhavyajeet](https://github.com/bhavyajeet)
