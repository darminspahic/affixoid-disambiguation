#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
feature_extractor

Short description:
This module writes various statistics about the affixoids_inventory.

License: MIT License
Version: 1.0

"""
import sys
import duden
from bs4 import BeautifulSoup
import requests

DATA_FILES_PATH = '../../data/final/'
FEATURES_OUTPUT_PATH = '../../data/features/'
RESSOURCES_PATH = '../../res/'


def create_affixoid_inventory(affixoid_file, class_name):
    dictionary = {}
    counter = 0
    with open(affixoid_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split()
            dict_key = word[-3]
            dictionary.update({dict_key: counter})

    for key in dictionary.keys():
        with open(affixoid_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                if key in word[-3] and class_name in word[-1]:
                    counter += 1
                    dictionary.update({key: counter})
        counter = 0

    return dictionary


def transform_to_binary(class_name):
    if class_name == 'Y':
        return 1
    if class_name == 'N':
        return 0
    else:
        print('Class Label not known. Exiting program')
        sys.exit()


def split_word_at_pipe(word):
    if '|' in word:
        return word.split('|')
    else:
        return [word, word]


def search_duden_frequency(words_inventory):
    if type(words_inventory) != list:
        words_inventory = words_inventory.split()

    words_inventory = replace_umlauts(words_inventory)
    frequency_list = []

    for w in words_inventory:
        words = duden.get(w)
        if words:
            print('Got word if: ', words)
            try:
                frequency_list.append(words.frequency)
            except AttributeError:
                frequency_list.append(0)
        else:
            first_word = get_first_result(w)
            words = duden.get(first_word)
            print('Got word else: ', words)
            try:
                frequency_list.append(words.frequency)
            except AttributeError:
                frequency_list.append(0)
    return frequency_list


def get_first_result(word):
    duden_url = 'http://www.duden.de/suchen/dudenonline/'
    r = requests.get(duden_url + word)
    data = r.text
    soup = BeautifulSoup(data, 'html.parser')
    try:
        main_sec = soup.find('section', id='block-duden-tiles-0')
        a_tags = [h2.a for h2 in main_sec.find_all('h2')]
        # print(a_tags[0].text)
        if a_tags[0].text == word:
            return a_tags[0].get('href').split('/')[-1]
        else:
            return 0
    except AttributeError:
        return 0


# needed for duden module
def replace_umlauts(word_list):
    umlaute = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue', 'ß': 'ss'}
    if type(word_list) == list:
        new_list = []
        for word in word_list:
            no_umlaut = word.translate({ord(k): v for k, v in umlaute.items()})
            new_list.append(no_umlaut)

        if len(word_list) == len(new_list):
            return new_list
        else:
            print('List error')
    if type(word_list) == str:
        return word_list.translate({ord(k): v for k, v in umlaute.items()})
    else:
        print('Replace Umlauts works only on strings and lists')


def get_dictionary_frequency(y_dict, n_dict, word, word_class):
    total = sum(y_dict.values()) + sum(n_dict.values())
    if word_class == 'Y':
        return total / y_dict[word]
    if word_class == 'N':
        return total / n_dict[word]
    else:
        print('Class not known')


def extract_frequency(input_file, frequency_file):
    input_inventory = []
    output_inventory = []
    with open(input_file, 'r', encoding='utf-8') as f, open(frequency_file, 'r', encoding='utf-8') as f2:
        lines1 = f.readlines()
        lines2 = f2.readlines()
        print(zip(f, f2))
        if len(lines1) == len(lines2):
            print('Files match by lines')
        for input_line, frequency_line in zip(f, f2):
            input_word = input_line.strip().split()
            frequency_word = frequency_line.strip().split()
            w_1 = input_word[0]
            w_2 = split_word_at_pipe(input_word[1])
            w_3 = frequency_word[0]
            input_inventory.append([w_1, w_2[0], w_2[1], w_3])

    # for i in input_inventory:
    #     if len(i) != 3:
    #         print('error')

    print(input_inventory)
    print(len(input_inventory))


if __name__ == "__main__":
    # extract_frequency(DATA_FILES_PATH + 'binary_unique_instance_prefixoid_segmentations.txt', DATA_FILES_PATH + 'binary_unique_instance_prefixoid_segmentations.txt')
    from sklearn.feature_extraction.text import CountVectorizer

    corpus = [
        'All my cats in a row',
        'When my cat sits down, she looks like a Furby toy!',
        'The cat from outer space',
        'Sunshine loves to sit like this for some reason.'
    ]

    vectorizer = CountVectorizer()
    print(vectorizer.fit_transform(corpus).todense())
    print(vectorizer.vocabulary_)

    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn import svm

    iris = datasets.load_iris()
    iris.data.shape, iris.target.shape

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=0)

    X_train.shape, y_train.shape

    X_test.shape, y_test.shape

    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    print(clf.score(X_test, y_test))