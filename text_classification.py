import numpy as np
# Import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from prettytable import PrettyTable
from sklearn.model_selection import cross_validate

# Load the corpus
from sklearn.datasets import load_files
corpus = load_files("corpus2/", encoding="utf-8)")
X, y = corpus.data, corpus.target # Assign the data to X, assign the labels to y

# Initialize CountVectorizer and convert documents into numerical features 
from sklearn.feature_extraction.text import CountVectorizer
vect1 = CountVectorizer(max_features=1500, min_df=5, max_df=0.9) # n=1
X1 = vect1.fit_transform(X).toarray()
vect2 = CountVectorizer(max_features=1500, min_df=5, max_df=0.9, ngram_range=(2,2)) # n=2
X2 = vect2.fit_transform(X).toarray()
vect3 = CountVectorizer(max_features=1500, min_df=5, max_df=0.9, ngram_range=(3,3)) # n=3
X3 = vect3.fit_transform(X).toarray()
vect4 = CountVectorizer(max_features=1500, min_df=5, max_df=0.9, ngram_range=(4,4)) # n=4
X4 = vect4.fit_transform(X).toarray()

features = [X1,X2,X3,X4]

# Table for the results
table = PrettyTable(["Classifier", "n", "Accuracy", "F1-Score"])

def classify(index, classifier, classifierName, n):
    scores = cross_validate(classifier, features[index], y, scoring=["accuracy","f1"], cv=5)
    accuracy = np.mean(scores["test_accuracy"])
    f1 = np.mean(scores["test_f1"])
    table.add_row([classifierName, n, round(accuracy, 3), round(f1, 3)])
    if index == 3:
        table.add_row(["","","",""])

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
for x in range(4):
    classify(x, lr, "Logistic Regression", x+1)

# Random Forest
rf = RandomForestClassifier(n_estimators=500, random_state=0)
for x in range(4):
    classify(x, rf, "Random Forest", x+1)

# Naive Bayes
nb = GaussianNB()
for x in range(4):
    classify(x, nb, "Naive Bayes", x+1)

# Support Vector Machine
svm = SVC(kernel="linear")
for x in range(4):
    classify(x, svm, "Support Vector Machine", x+1)

print("Classification results with only one type of n-gram:")
print(table)

# Another tables with a range of n-grams
table = PrettyTable(["Classifier", "n", "Accuracy", "F1-Score"])

vect1_2 = CountVectorizer(max_features=2000, ngram_range=(1,2), min_df=5, max_df=0.9) # n=1-2
X1_2 = vect1_2.fit_transform(X).toarray()
vect1_3 = CountVectorizer(max_features=2000, ngram_range=(1,3), min_df=5, max_df=0.9) # n=1-3
X1_3 = vect1_3.fit_transform(X).toarray()
vect1_4 = CountVectorizer(max_features=2000, ngram_range=(1,4), min_df=5, max_df=0.9) # n=1-4
X1_4 = vect1_4.fit_transform(X).toarray()
vect1_5 = CountVectorizer(max_features=2000, ngram_range=(1,5), min_df=5, max_df=0.9) # n=1-5
X1_5 = vect1_5.fit_transform(X).toarray()

features = [X1_2, X1_3, X1_4, X1_5]

# Logistic Regression
for x in range(4):
    classify(x, lr, "Logistic Regression", "1-{}".format(x+2))

# Random Forest
for x in range(4):
    classify(x, rf, "Random Forest", "1-{}".format(x+2))

# Naive Bayes
for x in range(4):
    classify(x, nb, "Naive Bayes", "1-{}".format(x+2))

# Support Vector Machine
for x in range(4):
    classify(x, svm, "Support Vector Machine", "1-{}".format(x+2))

print("Classification results with range of n-grams:")
print(table)

vect2_3 = CountVectorizer(max_features=2000, ngram_range=(2,3), min_df=5, max_df=0.9) #n=2-3
X2_3 = vect2_3.fit_transform(X).toarray()
vect2_4 = CountVectorizer(max_features=2000, ngram_range=(2,4), min_df=5, max_df=0.9) #n=2-4
X2_4 = vect2_4.fit_transform(X).toarray()
vect2_5 = CountVectorizer(max_features=2000, ngram_range=(2,5), min_df=5, max_df=0.9) #n=2-5
X2_5 = vect2_5.fit_transform(X).toarray()

features = [X2_3, X2_4, X2_5]
table = PrettyTable(["Classifier", "n", "Accuracy", "F1-Score"])

# Logistic Regression
for x in range(3):
    classify(x, lr, "Logistic Regression", "2-{}".format(x+3))

# Random Forest
for x in range(3):
    classify(x, rf, "Random Forest", "2-{}".format(x+3))

# Naive Bayes
for x in range(3):
    classify(x, nb, "Naive Bayes", "2-{}".format(x+3))

# Support Vector Machine
for x in range(3):
    classify(x, svm, "Support Vector Machine", "2-{}".format(x+3))

print("Classification results with range of n-grams:")
print(table)