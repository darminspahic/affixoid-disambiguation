#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
Classifier

Short description:
TODO
This module...

License: MIT License
Version: 1.0

"""

import sys
import numpy as np
import ast
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

################
# PATH SETTINGS
################
DATA_FEATURES_PATH = '../data/features/'


class Classifier:
    """ TODO

        Returns: Results

        Example: PREF = Classifier()

    """

    def __init__(self, string):
        print('=' * 40)
        print("Running Classifier on:", string)
        print('-' * 40)

    def read_features_from_files(self, feature_files_list):
        """TODO"""

        feature_instances = []
        files = []

        for file in feature_files_list:
            f = open(DATA_FEATURES_PATH+file, 'r', encoding='utf-8')
            files.append(f)

        zipped_files = zip(*files)

        for line in zipped_files:
            feature_vector = []
            for t in line:
                vec = t.split()
                for v in vec:
                    item = ast.literal_eval(v)  # evaluate type
                    feature_vector.append(item)
            feature_instances.append(feature_vector)

        return feature_instances

    def read_labels_from_file(self, file):
        """"""

        labels = []

        with open(DATA_FEATURES_PATH+file, 'r', encoding='utf-8') as feat_1:
            for line in feat_1:
                item = ast.literal_eval(line)
                labels.append(item)

        return labels

    def leave_one_out(self, feature_files_list, leave_out=None):
        """ TODO
            Leave out 0-9

        """

        files = []

        all_instances = []
        test_instances = []
        test_labels = []

        for file in feature_files_list:
            f = open(DATA_FEATURES_PATH + file, 'r', encoding='utf-8')
            files.append(f)

        zipped_files = zip(*files)

        for line in zipped_files:
            feature_vector = []
            for t in line:
                vec = t.split()
                for v in vec:
                    item = ast.literal_eval(v)  # evaluate type
                    feature_vector.append(item)
            all_instances.append(feature_vector)

        for item in all_instances:
            if item[leave_out] == 1:
                test_instances.append(item)
                test_labels.append(item[10])
        return test_instances, test_labels


class Style:
    BOLD = '\033[1m'
    END = '\033[0m'


if __name__ == "__main__":
    """ Features 
        'f0_' = candidate vector
        'f1_' = binary indicator, if affixoid

        'f2_' = frequency of complex word
        'f3_' = frequency of first part
        'f4_' = frequency of second part

        'f5_' = cosine similarity between complex word and affix

        'f6_' = vector of GermaNet supersenses for complex word
        'f7_' = vector of GermaNet supersenses for first part
        'f8_' = vector of GermaNet supersenses for second part

        'f9_'  = SentiMerge Polarity for complex word
        'f10_' = SentiMerge Polarity for first part
        'f11_' = SentiMerge Polarity for second part

        'f12_' = Affective Norms for complex word
        'f13_' = Affective Norms for first part
        'f14_' = Affective Norms for second part

        'f15_' = Emotion for complex word
        'f16_' = Emotion for first part
        'f17_' = Emotion for second part
    """

    PREF = Classifier('Prefixoids')
    SUFF = Classifier('Suffixoids')

    pref_X = PREF.read_features_from_files(['f0_pref.txt', 'f1_pref.txt', 'f2_pref.txt',
                                            'f3_pref.txt', 'f4_pref.txt', 'f5_pref.txt',
                                            'f6_pref.txt', 'f7_pref.txt', 'f8_pref.txt',
                                            'f9_pref.txt', 'f10_pref.txt', 'f11_pref.txt',
                                            'f12_pref.txt', 'f13_pref.txt', 'f14_pref.txt',
                                            'f15_pref.txt', 'f15_pref.txt', 'f17_pref.txt'])

    suff_X = SUFF.read_features_from_files(['f0_suff.txt', 'f1_suff.txt', 'f2_suff.txt',
                                            'f3_suff.txt', 'f4_suff.txt', 'f5_suff.txt',
                                            'f6_suff.txt', 'f7_suff.txt', 'f8_suff.txt',
                                            'f9_suff.txt', 'f10_suff.txt', 'f11_suff.txt',
                                            'f12_suff.txt', 'f13_suff.txt', 'f14_suff.txt',
                                            'f15_suff.txt', 'f15_suff.txt', 'f17_suff.txt'])

    """ Labels """
    pref_y = PREF.read_labels_from_file('f1_pref.txt')
    suff_y = SUFF.read_labels_from_file('f1_suff.txt')

    """ Split data """
    X_train_pref, X_test_pref, y_train_pref, y_test_pref = train_test_split(pref_X, pref_y, test_size=0.3, random_state=5, shuffle=True)
    X_train_suff, X_test_suff, y_train_suff, y_test_suff = train_test_split(suff_X, suff_y, test_size=0.3, random_state=5, shuffle=True)

    """ SVM """
    clf_pref = svm.SVC(gamma=0.001, C=10).fit(X_train_pref, y_train_pref)
    clf_suff = svm.SVC(gamma=0.001, C=10).fit(X_train_suff, y_train_suff)

    # clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    # clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False).fit(X_train, y_train)
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 1), random_state=1).fit(X_train, y_train)

    results_pref = clf_pref.predict(X_test_pref)
    results_suff = clf_suff.predict(X_test_suff)

    """ SCORES """
    def print_scores(classifier, classifer_results, test_instances, test_labels):
        print()
        print(Style.BOLD + 'RESULTS' + Style.END)
        print(classifer_results)
        print()
        print(Style.BOLD + 'SCORES' + Style.END)
        print('Classifier score: ', classifier.score(test_instances, test_labels))
        print('Precision: ', precision_score(test_labels, classifer_results))
        print('Recall: ', recall_score(test_labels, classifer_results))
        print('Average P-R score: ', average_precision_score(test_labels, classifer_results))
        print('F-1 Score: ', f1_score(test_labels, classifer_results, average='weighted'))
        print()

    print_scores(clf_pref, results_pref, X_test_pref, y_test_pref)
    print_scores(clf_suff, results_suff, X_test_suff, y_test_suff)

    def cross_validate(clf, instances, labels):
        cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=5)
        scores = cross_val_score(clf, instances, labels, cv=cv)
        print('CV scores:', scores)
        print()

    cross_validate(clf_pref, pref_X, pref_y)
    cross_validate(clf_suff, suff_X, suff_y)

    # ----------------------------



    # predicted = cross_val_predict(clf, suff_X, suff_y, cv=10)
    # print(metrics.accuracy_score(suff_y, predicted))

