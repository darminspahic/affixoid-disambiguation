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
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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


class Style:
    BOLD = '\033[1m'
    END = '\033[0m'


if __name__ == "__main__":
    """
        PREFIXOIDS
    """
    PREF = Classifier('Prefixoids')

    """ Features 
        'f0_pref.txt' = candidate vector
        'f1_pref.txt' = binary indicator, if affixoid
        'f2_pref.txt' = frequency of complex word
        'f3_pref.txt' = frequency of first part
        'f4_pref.txt' = frequency of second part
        'f5_pref.txt' = cosine similarity between complex word and affix
        'f6_pref.txt' = vector of GermaNet supersenses for complex word
        'f7_pref.txt' = vector of GermaNet supersenses for first part
        'f8_pref.txt' = vector of GermaNet supersenses for second part
        'f9_pref.txt' = SentiMerge Polarity for complex word
        'f10_pref.txt' = SentiMerge Polarity for first part
        'f11_pref.txt' = SentiMerge Polarity for second part
    """
    pref_X = PREF.read_features_from_files(['f0_pref.txt', 'f1_pref.txt', 'f2_pref.txt',
                                            'f3_pref.txt', 'f4_pref.txt', 'f5_pref.txt',
                                            'f6_pref.txt', 'f7_pref.txt', 'f8_pref.txt',
                                            'f9_pref.txt', 'f10_pref.txt', 'f11_pref.txt'])

    """ Labels """
    pref_y = PREF.read_labels_from_file('f1_pref.txt')

    """ Split data """
    X_train, X_test, y_train, y_test = train_test_split(pref_X, pref_y, test_size=0.3, random_state=5, shuffle=True)

    """ SVM """
    # clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    clf = svm.SVC(gamma=0.001, C=100).fit(X_train, y_train)
    results = clf.predict(X_test)

    """ SCORES """
    print()
    print(Style.BOLD + 'RESULTS' + Style.END)
    print(results)
    print()
    print(Style.BOLD + 'SCORES' + Style.END)
    print('Classifier score: ', clf.score(X_test, y_test))
    print('Precision: ', precision_score(y_test, results))
    print('Recall: ', recall_score(y_test, results))
    print('Average P-R score: ', average_precision_score(y_test, results))
    print('F-1 Score: ', f1_score(y_test, results, average='weighted'))
