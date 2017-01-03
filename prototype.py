#!/usr/bin/python

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt
import operator
import time
import numpy as np
from dataset import DataSet
from collections import Counter
from random import randint

NUM_OF_COPIES = 10
NUM_OF_CHANGED_FEATURES = 2


def generate_test_set():
    input_file = open("data_sets/flags_clean.csv", "r", newline='', encoding="utf8")
    output_file = open("data_sets/test_set.csv", "w", newline='', encoding="utf8")
    output_file.write(input_file.readline())
    for line in input_file:
        for _1 in range(NUM_OF_COPIES):
            features = line.split(",")
            features_to_change = []
            for _2 in range(NUM_OF_CHANGED_FEATURES):
                features_to_change.append(randint(1, len(features) - 1))
            for feature_id in range(len(features)):
                if feature_id in features_to_change:
                    if int(features[feature_id]) > 0:
                        features[feature_id] = '0'
                    else:
                        features[feature_id] = '1'
                    if feature_id == len(features) - 1:
                        features[feature_id] += '\n'
            joined = ','.join(str(feature) for feature in features)
            output_file.write(joined)


def select_alpha(ds, hidden_layer_size):
    # The most effective alpha value selector:
    for alpha in np.arange(0.1, 1.0, 0.005):
        clf = MLPClassifier(algorithm='l-bfgs', max_iter=500, alpha=alpha, hidden_layer_sizes=hidden_layer_size, random_state=1)
        # classifier = clf.fit(X_train, y_train)
        clf.fit(ds.X, ds.y)
        print("Alpha: {:.3}".format(alpha))
        correct = 0
        for i in range(len(ds.X)):
            if clf.predict([ds.X[i]]) == ds.y[i]:
                correct += 1
        print("Number of correct answers: {}, effectiveness: {:.3f}%".format(correct, float(correct / len(ds.y))))


def select_hidden_layer_size(ds, alpha):
    # The most effective hidden layer size selector:
    output = {}
    times = {}
    for hidden_size in np.arange(1, 11, 1):
        clf = MLPClassifier(algorithm='l-bfgs', max_iter=500, alpha=alpha, hidden_layer_sizes=hidden_size, random_state=1)
        start = time.time()
        clf.fit(ds.X, ds.y)
        stop = time.time()
        times[hidden_size] = stop - start
        print("Hidden layer size: {}, fit time: {} s".format(hidden_size, stop - start))
        correct = 0
        for i in range(len(ds.X)):
            predicted = clf.predict([ds.X[i]])
            if predicted == ds.y[i]:
                correct += 1
        effectiveness = float(correct / len(ds.y)) * 100
        print("Number of correct answers: {}, effectiveness: {:.3f}%".format(correct, effectiveness))
        output[hidden_size] = effectiveness
    return output, times


def select_hidden_layer_sizes(ds, alpha):
    # The most effective hidden layers size selector:
    for hid1 in np.arange(1, 25, 1):
        for hid2 in np.arange(1, 25, 1):
            clf = MLPClassifier(algorithm='l-bfgs', max_iter=500, alpha=alpha, hidden_layer_sizes=(hid1, hid2), random_state=1)
            clf.fit(ds.X, ds.y)
            print("1st size: {}, 2nd size: {}".format(hid1, hid2))
            correct = 0
            for i in range(len(ds.X)):
                predicted = clf.predict([ds.X[i]])
                if predicted == ds.y[i]:
                    correct += 1
            print("Number of correct answers: {}, effectiveness: {:.3f}%".format(correct, float(correct / len(ds.y))))


def test_effectiveness(ds, clf):
    # Flag guesser effectiveness tester:
    correct = 0
    for i in range(len(ds.X)):
        #if ds.y[i] == "Niger":
        #    print("Top kek")
        predicted = clf.predict([ds.X[i]])
        probabilities = clf.predict_proba([ds.X[i]])
        countries_probabilities = {}
        for k in range(len(clf.classes_)):
            countries_probabilities[clf.classes_[k]] = probabilities[0][k]

        sorted_probabilities = sorted(countries_probabilities.items(), key=operator.itemgetter(1), reverse=True)
        print("------------------------------------------------")
        print("Country: {}".format(ds.y[i]))
        p1 = sorted_probabilities[0]
        p2 = sorted_probabilities[1]
        p3 = sorted_probabilities[2]
        print("1st probability: {:.3f}% of country: {}".format(p1[1], p1[0]))
        print("2nd probability: {:.3f}% of country: {}".format(p2[1], p2[0]))
        print("3rd probability: {:.3f}% of country: {}".format(p3[1], p3[0]))

        if predicted == ds.y[i]:
            correct += 1
        else:
            print("WRONG prediction - country: {}, predicted as: {}".format(ds.y[i], predicted[0]))
    print("Number of correct answers: {}, total number of guesses: {} effectiveness: {:.3f}%".format(correct, len(ds.y), float(correct / len(ds.y))))


def select_best_features(ds):
    # Best features selector:
    counter = Counter()
    for n_features in range(1, ds.number_of_features, 1):
        best_features = SelectKBest(k=n_features).fit(ds.X, ds.y).get_support(indices=True)
        counter.update(best_features)
    #labels, values = zip(*counter.items())
    #print(labels)
    #print(values)
    for label, value in sorted(counter.items(), key=operator.itemgetter(1), reverse=True):
        print("{} : {}".format(ds.col_names[label], value), end=", ")
    print('\n')
    return counter


def plot_best_features():
    new_set = DataSet()
    general_counter = Counter()
    for i in range(20):
        generate_test_set()
        with open("data_sets/test_set.csv",
              "r", newline='', encoding="utf8") as csv_file:
            new_set.extract_from_csv(csv_file)
        general_counter.update(select_best_features(new_set))
    print("General counter:")
    labels = []
    values = []
    for label, value in sorted(general_counter.items(), key=operator.itemgetter(1), reverse=True):
        labels.append(new_set.col_names[label])
        values.append(value)
        print("{} : {}".format(new_set.col_names[label], value), end=", ")
    print('\n')
    indexes = np.arange(len(values))
    plt.bar(indexes, values, width=1.0, align='center')
    plt.subplots_adjust(bottom=0.20)
    plt.xlim(-0.5, len(labels) - 0.5)
    plt.xticks(indexes, labels, rotation='vertical')
    plt.xlabel("Features")
    plt.ylabel("Features selection")
    plt.show()


def plot_hidden_layer_sizes(ds, alpha):
    hidden_sizes, hidden_times = select_hidden_layer_size(ds, alpha)
    keys, values = hidden_sizes.keys(), hidden_sizes.values()
    plt.bar(list(keys), values, width=1.0, align='center')
    plt.xlim(0.5, len(list(keys)) + 0.5)
    plt.xticks(list(keys))
    plt.xlabel("Size of neurons in hidden layer")
    plt.ylabel("Score (%)")
    plt.show()

    layers, times = hidden_times.keys(), hidden_times.values()
    plt.plot(list(layers), list(times), "bo--")
    plt.xticks(list(layers))
    plt.xlabel("Size of neurons in hidden layer")
    plt.ylabel("Time (s)")
    plt.show()


def main():
    ds = DataSet()
    with open("data_sets/flags_clean.csv",
              "r", newline='', encoding="utf8") as csv_file:
        ds.extract_from_csv(csv_file)

    hidden_layer_size = 8
    alpha = 0.88
    clf = MLPClassifier(algorithm='l-bfgs', max_iter=500, alpha=alpha, hidden_layer_sizes=hidden_layer_size, random_state=1)
    clf.fit(ds.X, ds.y)
    #print("Hidden layer size: {}".format(hidden_layer_size))

    plot_best_features()
    plot_hidden_layer_sizes(ds, alpha)
    '''
    test_set = DataSet()
    with open("data_sets/test_set.csv",
              "r", newline='', encoding="utf8") as csv_file:
        test_set.extract_from_csv(csv_file)
    test_effectiveness(test_set, clf)
    '''
    #select_best_features(ds)
    #select_alpha(ds, hidden_layer_size)
    #select_hidden_layer_sizes(ds, alpha)
    #select_hidden_layer_size(ds, alpha)


if __name__ == '__main__':
    main()
