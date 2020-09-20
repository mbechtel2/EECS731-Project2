################################################################################
#
# File: project2
# Author: Michael Bechtel
# Date: September 21, 2020
# Class: EECS 731
# Description: Use classification models to determine the Player in a given
#               Shakespeare play using features such as the Play and ActSceneLine.
# 
################################################################################

import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Create all classification models
class_NaiveBayes = GaussianNB()
class_DecisionTree = tree.DecisionTreeClassifier()
class_RandomForest = RandomForestClassifier(max_depth=6)
class_NeuralNetwork = MLPClassifier(max_iter=1000)

# Read the entire dataset
player_dataset = pd.read_csv("../data/raw/Shakespeare_data.csv")

################################################################################

# Base case - use all features
print()
print("======================================")
print("Base case: All Features => Player")
print("======================================")

# Retrieve all columns from the dataset
column_names = ["Play", "ActSceneLine", "PlayerLine", "PlayerLinenumber", "Player"]
columns = player_dataset.loc[:,column_names].dropna()
columns = columns.drop_duplicates()
columns.to_csv("../data/processed/base-case.csv")

# Extract the features and players
feature_names = column_names[:-1]
features = columns.loc[:,feature_names]
players_name = column_names[-1]
players = columns.loc[:,players_name]

# Encode all columns
encoder = preprocessing.LabelEncoder()
for f in feature_names:
    features[f] = encoder.fit_transform(features[f])
players = encoder.fit_transform(players)

# Create train and test sets
features_train, features_test, players_train, players_test = train_test_split(features, players, test_size=0.50, random_state=0, shuffle=True)

# Run all classification models and display their respective results.
players_pred = class_NaiveBayes.fit(features_train, players_train).predict(features_test)
print("Naive Bayes Accuracy: {:.2f}".format(accuracy_score(players_test, players_pred)))

players_pred = class_DecisionTree.fit(features_train, players_train).predict(features_test)
print("Decision Tree Accuracy: {:.2f}".format(accuracy_score(players_test, players_pred)))

players_pred = class_RandomForest.fit(features_train, players_train).predict(features_test)
print("Random Forest Accuracy: {:.2f}".format(accuracy_score(players_test, players_pred)))

players_pred = class_NeuralNetwork.fit(features_train, players_train).predict(features_test)
print("Neural Network Accuracy: {:.2f}".format(accuracy_score(players_test, players_pred)))

################################################################################

# Subset case 1 - only use the Play and ActSceneLine features
print()
print("======================================")
print("Set 1: Play + ActSceneLine => Player")
print("======================================")

# Retrieve the desired columns from the dataset
column_names = ["Play", "ActSceneLine", "Player"]
columns = player_dataset.loc[:,column_names].dropna()
columns = columns.drop_duplicates()
columns.to_csv("../data/processed/subset-case-1.csv")

# Extract the features and players
feature_names = column_names[:-1]
features = columns.loc[:,feature_names]
players_name = column_names[-1]
players = columns.loc[:,players_name]

# Encode all columns
encoder = preprocessing.LabelEncoder()
for f in feature_names:
    features[f] = encoder.fit_transform(features[f])
players = encoder.fit_transform(players)

# Create train and test sets
features_train, features_test, players_train, players_test = train_test_split(features, players, test_size=0.50, random_state=0, shuffle=True)

# Run all classification models and display their respective results.
players_pred = class_NaiveBayes.fit(features_train, players_train).predict(features_test)
print("Naive Bayes Accuracy: {:.2f}".format(accuracy_score(players_test, players_pred)))

players_pred = class_DecisionTree.fit(features_train, players_train).predict(features_test)
print("Decision Tree Accuracy: {:.2f}".format(accuracy_score(players_test, players_pred)))

players_pred = class_RandomForest.fit(features_train, players_train).predict(features_test)
print("Random Forest Accuracy: {:.2f}".format(accuracy_score(players_test, players_pred)))

players_pred = class_NeuralNetwork.fit(features_train, players_train).predict(features_test)
print("Neural Network Accuracy: {:.2f}".format(accuracy_score(players_test, players_pred)))

################################################################################

# Subset case 1 - only use the PlayerLine and PlayerLinenumber features
print()
print("======================================")
print("Set 2: PlayerLine + Playerlinenumber => Player")
print("======================================")

# Retrieve the desired columns from the dataset
column_names = ["PlayerLine", "PlayerLinenumber", "Player"]
columns = player_dataset.loc[:,column_names].dropna()
columns = columns.drop_duplicates()
columns.to_csv("../data/processed/subset-case-2.csv")

# Extract the features and players
feature_names = column_names[:-1]
features = columns.loc[:,feature_names]
players_name = column_names[-1]
players = columns.loc[:,players_name]

# Encode all columns
encoder = preprocessing.LabelEncoder()
for f in feature_names:
    features[f] = encoder.fit_transform(features[f])
players = encoder.fit_transform(players)

# Create train and test sets
features_train, features_test, players_train, players_test = train_test_split(features, players, test_size=0.50, random_state=0, shuffle=True)

# Run all classification models and display their respective results.
players_pred = class_NaiveBayes.fit(features_train, players_train).predict(features_test)
print("Naive Bayes Accuracy: {:.2f}".format(accuracy_score(players_test, players_pred)))

players_pred = class_DecisionTree.fit(features_train, players_train).predict(features_test)
print("Decision Tree Accuracy: {:.2f}".format(accuracy_score(players_test, players_pred)))

players_pred = class_RandomForest.fit(features_train, players_train).predict(features_test)
print("Random Forest Accuracy: {:.2f}".format(accuracy_score(players_test, players_pred)))

players_pred = class_NeuralNetwork.fit(features_train, players_train).predict(features_test)
print("Neural Network Accuracy: {:.2f}".format(accuracy_score(players_test, players_pred)))
