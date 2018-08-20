#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
Classifier

Short description:
Classifier module

License: MIT License
Version: 1.0

"""

import configparser
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier

from modules import file_reader as fr

########################
# GLOBAL FILE SETTINGS
########################
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('config.ini')


def grid_search_report(X_train, y_train, X_test, y_test, affixoid_class):
    """ This function implements the GridSearch algorithm described in :mod:`sklearn.model_selection._search` """

    import warnings
    warnings.filterwarnings("ignore")

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-5, 1e-4, 1e-3, 1e-2, 0.1], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set for:", affixoid_class)
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


def cross_validate(clf, instances, labels):
    """ Crossvalidation function """

    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=5)
    scores = cross_val_score(clf, instances, labels, cv=cv)
    print('5-Fold crossvalidation:', scores)


def print_scores(score_title, classifier, classifer_results, test_instances, test_labels):
    """ Helper function for printing scores """

    import warnings
    warnings.filterwarnings("ignore")

    print()
    print(Style.BOLD + 'SCORES', score_title + Style.END)
    print('Classifier score: ', classifier.score(test_instances, test_labels))
    print()
    target_names = ['affixoid', 'non-affixoid']
    print(classification_report(test_labels, classifer_results, target_names=target_names))
    print('\nConfusion matrix:')
    print(confusion_matrix(test_labels, classifer_results))
    print()


def plot(y_test, y_score):
    """
    Example taken from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    """
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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Example taken from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


class Style:
    """ Helper class for nicer coloring """
    BOLD = '\033[1m'
    END = '\033[0m'


if __name__ == "__main__":
    """ Files for binary classification
    
        Labels:
        'f1_' = binary indicator if affixoid; labels

        Features:
        'f0_' = candidate vector; not used in the experiments

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

    """ Read features from files """
    pref_X = fr.read_features_from_files(['f2_pref.txt', 'f3_pref.txt', 'f4_pref.txt',
                                            'f5_pref.txt',
                                            'f6_pref.txt', 'f7_pref.txt', 'f8_pref.txt',
                                            'f9_pref.txt', 'f10_pref.txt', 'f11_pref.txt',
                                            'f12_pref.txt', 'f13_pref.txt', 'f14_pref.txt',
                                            'f15_pref.txt', 'f16_pref.txt', 'f17_pref.txt',
                                            'f18_pref.txt'
                                            ], path=config.get('PathSettings', 'DataFeaturesPath'))

    suff_X = fr.read_features_from_files(['f2_suff.txt', 'f3_suff.txt', 'f4_suff.txt',
                                            'f5_suff.txt',
                                            'f6_suff.txt', 'f7_suff.txt', 'f8_suff.txt',
                                            'f9_suff.txt', 'f10_suff.txt', 'f11_suff.txt',
                                            'f12_suff.txt', 'f13_suff.txt', 'f14_suff.txt',
                                            'f15_suff.txt', 'f16_suff.txt', 'f17_suff.txt',
                                            'f18_suff.txt'
                                            ], path=config.get('PathSettings', 'DataFeaturesPath'))

    """ Scale data """
    scaler_s = preprocessing.StandardScaler()
    scaler_m = preprocessing.MinMaxScaler()
    scaler_r = preprocessing.RobustScaler()

    pref_X_scaled = scaler_m.fit_transform(pref_X)
    suff_X_scaled = scaler_m.fit_transform(suff_X)

    # print(pref_X[0])
    # print(suff_X[0])
    # print(pref_X_scaled[0])
    # print(suff_X_scaled[0])

    """ Labels """
    pref_y = fr.read_labels_from_file('f1_pref.txt', path=config.get('PathSettings', 'DataFeaturesPath'))
    suff_y = fr.read_labels_from_file('f1_suff.txt', path=config.get('PathSettings', 'DataFeaturesPath'))

    """ Split data """
    X_train_pref, X_test_pref, y_train_pref, y_test_pref = train_test_split(pref_X_scaled, pref_y, test_size=0.3, random_state=5, shuffle=True)
    X_train_suff, X_test_suff, y_train_suff, y_test_suff = train_test_split(suff_X_scaled, suff_y, test_size=0.3, random_state=5, shuffle=True)

    """ Parameter Selection with GridSearch """
    # grid_search_report(X_train_pref, y_train_pref, X_test_pref, y_test_pref, 'Prefixoids')
    # grid_search_report(X_train_suff, y_train_suff, X_test_suff, y_test_suff, 'Suffixoids')

    """ SVM """
    clf_pref = svm.SVC(kernel="rbf", gamma=0.01, C=100).fit(X_train_pref, y_train_pref)  # GridSearch results: {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
    clf_suff = svm.SVC(kernel="rbf", gamma=0.1, C=10).fit(X_train_suff, y_train_suff)  # GridSearch results: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
    results_pref = clf_pref.predict(X_test_pref)
    results_suff = clf_suff.predict(X_test_suff)

    """ Most frequent sense classifier """
    clf_dummy_pref = DummyClassifier(strategy='most_frequent', random_state=0).fit(X_train_pref, y_train_pref)
    clf_dummy_suff = DummyClassifier(strategy='most_frequent', random_state=0).fit(X_train_suff, y_train_suff)
    results_pref_mfs = clf_dummy_pref.predict(X_test_pref)
    results_suff_mfs = clf_dummy_pref.predict(X_test_suff)

    """ Crossvalidation """
    """ SCORES """
    print_scores('Most frequent sense baseline for prefixoids', clf_dummy_pref, results_pref_mfs, X_test_pref, y_test_pref)
    print_scores('Prefixoids', clf_pref, results_pref, X_test_pref, y_test_pref)
    cross_validate(svm.SVC(kernel="rbf", gamma=0.01, C=100), pref_X_scaled, pref_y)

    print_scores('Most frequent sense baseline for suffixoids', clf_dummy_suff, results_suff_mfs, X_test_suff, y_test_suff)
    print_scores('Suffixoids', clf_suff, results_suff, X_test_suff, y_test_suff)
    cross_validate(svm.SVC(kernel="rbf", gamma=0.1, C=10), suff_X_scaled, suff_y)

    """ Plot precision recall and learning curve """
    # plot(y_test_pref, results_pref)
    # plot(y_test_suff, results_suff)

    # plot_learning_curve(clf_pref, 'Training data prefixoids', X_train_pref, y_train_pref, ylim=(0.65, 1.0), cv=5, n_jobs=4)
    # plot_learning_curve(clf_suff, 'Training data suffixoids', X_train_suff, y_train_suff, ylim=(0.65, 1.0), cv=5, n_jobs=4)

    """ 
        Scores for: Word Sense Disambiguation 
    """
    print()
    print('=' * 40)
    print(Style.BOLD + "Scores for Word Sense Disambiguation" + Style.END)
    print('-' * 40)

    pref_WSD_labels = fr.read_features_from_files(['f0_pref_wsd_final.txt'], path=config.get('PathSettings', 'DataWsdPath'))
    pref_WSD_scores = fr.read_features_from_files(['f1_pref_wsd_final.txt'], path=config.get('PathSettings', 'DataWsdPath'))

    suff_WSD_labels = fr.read_features_from_files(['f0_suff_wsd_final.txt'], path=config.get('PathSettings', 'DataWsdPath'))
    suff_WSD_scores = fr.read_features_from_files(['f1_suff_wsd_final.txt'], path=config.get('PathSettings', 'DataWsdPath'))

    print(Style.BOLD + 'WSD SCORES Prefixoids' + Style.END)
    print('Precision: ', precision_score(pref_WSD_labels, pref_WSD_scores))
    print('Recall: ', recall_score(pref_WSD_labels, pref_WSD_scores))
    print('F-1 Score: ', f1_score(pref_WSD_labels, pref_WSD_scores, average='weighted'))
    print('ROC AUC Score: ', roc_auc_score(pref_WSD_labels, pref_WSD_scores))
    print('\nConfusion matrix:')
    print(confusion_matrix(pref_WSD_labels, pref_WSD_scores))
    print()

    print(Style.BOLD + 'WSD SCORES Suffixoids' + Style.END)
    print('Precision: ', precision_score(suff_WSD_labels, suff_WSD_scores))
    print('Recall: ', recall_score(suff_WSD_labels, suff_WSD_scores))
    print('F-1 Score: ', f1_score(suff_WSD_labels, suff_WSD_scores, average='weighted'))
    print('ROC AUC Score: ', roc_auc_score(suff_WSD_labels, suff_WSD_scores))
    print('\nConfusion matrix:')
    print(confusion_matrix(suff_WSD_labels, suff_WSD_scores))
    print()
