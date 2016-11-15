#!/usr/bin/python

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
import random
from dataset import DataSet

def main():
    ds = DataSet()
    with open("data_sets/bycmoze.csv",
              "r", newline='', encoding="utf8") as csv_file:
        ds.extract_from_csv(csv_file)

    # X_train, X_test, y_train, y_test = train_test_split(ds.X, ds.y, test_size=0.3, random_state=1)
    clf = MLPClassifier(algorithm='l-bfgs', max_iter=500, alpha=1e-10, hidden_layer_sizes=10, random_state=1)
    # classifier = clf.fit(X_train, y_train)
    clf.fit(ds.X, ds.y)
    print(clf.classes_)
    print(clf.predict_proba([[0,3,5,1,1,0,1,1,1,0,3,0,0,0,0,1,0,0,1,0,0,7,3]])) # Afganistan
                       #[0,0,3,1,0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,1,0,1,1], # Albania
                       #[0,1,5,1,0,1,1,1,1,0,1,0,0,0,0,1,0,1,0,0,0,7,1], # Antigua costam
                       #[0,3,3,0,1,1,1,0,0,0,3,0,0,0,0,0,0,0,0,0,0,3,2]])) # Gabon


if __name__ == '__main__':
    main()

