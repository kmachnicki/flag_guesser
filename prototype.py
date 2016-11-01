#!/usr/bin/python

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
import random
from dataset import DataSet

def main():
    ds = DataSet()
    with open("data_sets/flag.csv",
              "r", newline='', encoding="utf8") as csv_file:
        ds.extract_from_csv(csv_file)

    # X_train, X_test, y_train, y_test = train_test_split(ds.X, ds.y, test_size=0.3, random_state=1)
    clf = MLPClassifier(algorithm='l-bfgs', max_iter=500, alpha=1e-5, hidden_layer_sizes=30, random_state=1)
    # classifier = clf.fit(X_train, y_train)
    clf.fit(ds.X, ds.y)
    print(clf.predict([[5,1,648,16,10,2,0,3,5,1,1,0,1,1,1,0,3,0,0,0,0,1,0,0,1,0,0,7,3], # Afganistan
                       [3,1,29,3,6,6,0,0,3,1,0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,1,0,1,1]])) # Albania

if __name__ == '__main__':
    main()

