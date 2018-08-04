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
import ast
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

################
# PATH SETTINGS
################
DATA_FEATURES_PATH = '../data/features/'
DATA_WSD_PATH = '../data/wsd/'


class Classifier:
    """ TODO

        Returns: Results

        Example: PREF = Classifier()

    """

    def __init__(self):
        print('-' * 40)

    def read_features_from_files(self, feature_files_list, path=DATA_FEATURES_PATH):
        """TODO"""

        feature_instances = []
        files = []

        for file in feature_files_list:
            f = open(path+file, 'r', encoding='utf-8')
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

    def read_labels_from_file(self, file, path=DATA_FEATURES_PATH):
        """"""

        labels = []

        with open(path+file, 'r', encoding='utf-8') as feat_1:
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
        
        'f18_' = PMI Scores for first and second part of word
    """

    PREF = Classifier()
    SUFF = Classifier()

    pref_X = PREF.read_features_from_files(['f2_pref.txt',
                                            'f3_pref.txt', 'f4_pref.txt', 'f5_pref.txt',
                                            'f6_pref.txt', 'f7_pref.txt', 'f8_pref.txt',
                                            'f9_pref.txt', 'f10_pref.txt', 'f11_pref.txt',
                                            'f12_pref.txt', 'f13_pref.txt', 'f14_pref.txt',
                                            'f15_pref.txt', 'f15_pref.txt', 'f17_pref.txt',
                                            'f18_pref.txt'])

    suff_X = SUFF.read_features_from_files(['f2_suff.txt',
                                            'f3_suff.txt', 'f4_suff.txt', 'f5_suff.txt',
                                            'f6_suff.txt', 'f7_suff.txt', 'f8_suff.txt',
                                            'f9_suff.txt', 'f10_suff.txt', 'f11_suff.txt',
                                            'f12_suff.txt', 'f13_suff.txt', 'f14_suff.txt',
                                            'f15_suff.txt', 'f15_suff.txt', 'f17_suff.txt',
                                            'f18_suff.txt'])



    scaler_s = preprocessing.StandardScaler()
    scaler_m = preprocessing.MinMaxScaler()
    scaler_r = preprocessing.RobustScaler()

    pref_X_scaled = scaler_m.fit_transform(pref_X)
    suff_X_scaled = scaler_m.fit_transform(suff_X)

    # print(pref_X_scaled[0])
    # print(suff_X_scaled[0])

    """ Labels """
    pref_y = PREF.read_labels_from_file('f1_pref.txt')
    suff_y = SUFF.read_labels_from_file('f1_suff.txt')

    """ Split data """
    X_train_pref, X_test_pref, y_train_pref, y_test_pref = train_test_split(pref_X_scaled, pref_y, test_size=0.3, random_state=5, shuffle=True)
    X_train_suff, X_test_suff, y_train_suff, y_test_suff = train_test_split(suff_X_scaled, suff_y, test_size=0.3, random_state=5, shuffle=True)

    """ SVM """
    clf_pref = svm.SVC(kernel="rbf", gamma=0.1, C=1).fit(X_train_pref, y_train_pref)
    clf_suff = svm.SVC(kernel="rbf", gamma=0.1, C=10).fit(X_train_suff, y_train_suff)

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
        cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=5)
        scores = cross_val_score(clf, instances, labels, cv=cv)
        print('Crossvalidation scores:', scores)

    cross_validate(clf_pref, pref_X_scaled, pref_y)
    cross_validate(clf_suff, suff_X_scaled, suff_y)

    print()

    def plot(y_test, y_score):
        average_precision = average_precision_score(y_test, y_score)
        precision, recall, _ = precision_recall_curve(y_test, y_score)

        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

        plt.show()

    # plot(y_test_pref, results_pref)
    # plot(y_test_suff, results_suff)

    # ---------------------------

    def svc_param_selection(X, y, nfolds):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma': gammas}
        grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
        grid_search.fit(X, y)
        # grid_search.best_params_
        return grid_search.best_params_

    # print(svc_param_selection(pref_X_scaled, pref_y, 10))
    # print(svc_param_selection(suff_X_scaled, suff_y, 10))

    """ 
        Word Sense Disambiguation 
    """

    pref_WSD_labels = PREF.read_features_from_files(['f0_pref_wsd_10.txt'], path=DATA_WSD_PATH)
    pref_WSD_scores = PREF.read_features_from_files(['f1_pref_wsd_10.txt'], path=DATA_WSD_PATH)

    suff_WSD_labels = SUFF.read_features_from_files(['f0_suff_wsd_10.txt'], path=DATA_WSD_PATH)
    suff_WSD_scores = SUFF.read_features_from_files(['f1_suff_wsd_10.txt'], path=DATA_WSD_PATH)

    print(Style.BOLD + 'WSD Scores Prefixoids' + Style.END)
    print('Precision: ', precision_score(pref_WSD_labels, pref_WSD_scores))
    print('Recall: ', recall_score(pref_WSD_labels, pref_WSD_scores))
    print('F-1 Score: ', f1_score(pref_WSD_labels, pref_WSD_scores, average='weighted'))
    print()

    print(Style.BOLD + 'WSD Scores Suffixoids' + Style.END)
    print('Precision: ', precision_score(suff_WSD_labels, suff_WSD_scores))
    print('Recall: ', recall_score(suff_WSD_labels, suff_WSD_scores))
    print('F-1 Score: ', f1_score(suff_WSD_labels, suff_WSD_scores, average='weighted'))
    print()