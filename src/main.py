#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
main

Short description:
This main module...

License: MIT License
Version: 1.0

"""

import sys
# import os
# import duden
# import requests
import numpy as np
import matplotlib.pyplot as plt
# from bs4 import BeautifulSoup

################
# PATH SETTINGS
################
DATA_PATH = '../data/'
DATA_FINAL_PATH = '../data/final/'
DATA_FEATURES_PATH = '../data/features/'

################
# FILE SETTINGS
################
FINAL_PREFIXOID_FILE = 'binary_unique_instance_prefixoid_segmentations.txt'
FINAL_SUFFIXOID_FILE = 'binary_unique_instance_suffixoid_segmentations.txt'
FREQUENCY_PREFIXOID_LEMMAS = 'prefixoid_lemmas_freqs.csv'
FREQUENCY_SUFFIXOID_LEMMAS = 'suffixoid_lemmas_freqs.csv'
FREQUENCY_PREFIXOID_HEADS = 'lemma_frequencies_unique_heads_of_prefixoid_formations.csv'
FREQUENCY_SUFFIXOID_MODIFIERS = 'modifiers_of_suffixoids_lemmas_freqs.csv'


class AffixoidClassifier:
    """ This is the main module and it is a collection of all
        modules from the project.

        Returns: Results from all modules of this project

        Example: PREFIXOIDS = AffixoidClassifier()

    """

    def __init__(self, string):
        print('=' * 40)
        print("Running AffixoidClassifier on:", string)
        print('-' * 40)

    def create_affixoid_inventory(self, affixoid_file, class_name):
        """ This function creates a dictionary with class instances of affixoids

            Args:
                affixoid_file (file): File with affixoid instances
                class_name (str): Class label

            Returns:
                Dictionary with class instances

            Example:
                >>> create_affixoid_inventory(DATA_FINAL_PATH+FINAL_PREFIXOID_FILE, 'Y')

        """
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

    def read_file_to_list(self, affixoid_file):
        """ This function reads a file with affixoids and returns a list of lines from file

            Args:
                affixoid_file (file): File with affixoid instances

            Returns:
                List of lines from file

            Example:
                >>> read_file_to_list(DATA_FINAL_PATH+FINAL_PREFIXOID_FILE)

        """
        file_as_list = []
        with open(affixoid_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                file_as_list.append(word)

        return file_as_list

    def transform_to_binary(self, class_name):
        """ This function transforms class labels to binary indicators

            Args:
                class_name (str): Class label

            Returns:
                Binary indicator for class label [0,1]

            Examples:
                >>> transform_to_binary('Y')

        """

        if class_name == 'Y':
            return 1
        if class_name == 'N':
            return 0
        else:
            print('Class Label not known. Exiting program')
            sys.exit()

    def split_word_at_pipe(self, word):
        """ This function splits a word separated by a | symbol

            Args:
                word (str): Word with a pipe symbol

            Returns:
                A list of split items

            Examples:
                >>> split_word_at_pipe('Bilderbuch|Absturz')

        """

        if '|' in word:
            return word.split('|')
        else:
            return [word, word]

    def extract_frequency(self, word, dictionary):
        """ This function extracts frequencies for a given word from a dictionary

            Args:
                word (str): Word
                dictionary (dict): Dictionary with frequencies

            Returns:
                A frequency for a given word from a dictionary

            Examples:
                >>> extract_frequency('Bilderbuch', {'Bilderbuch':30})

        """
        if word in dictionary.keys():
            value = dictionary[word]
            return value
        else:
            # TODO:
            # Ikone fehlt
            # Bhaishajya fehlt
            # Ostgoten fehlt
            # Splatterfilm fehlt
            return word

    def create_frequency_dictionary(self, frequency_file):
        """ This function creates a dictionary with frequency instances of affixoids

            Args:
                frequency_file (file): File with affixoid frequencies

            Returns:
                Dictionary with frequency instances of affixoids

            Example:
                >>> create_frequency_dictionary(DATA_PATH+FREQUENCY_PREFIXOID_LEMMAS)

        """

        dictionary = {}
        with open(frequency_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                dict_key = word[0]
                dict_value = word[1]
                dictionary.update({dict_key: dict_value})

        return dictionary

    def plot_statistics(self, dict_1, dict_2, title):
        """ This function plots charts with affixoid statistics.

            Args:
                dict_1 (dict): Dictionary with Y instances
                dict_2 (dict): Dictionary with N instances
                title (str): Title of the chart

            Returns:
                Matplotlib images

            Example:
                >>> plot_statistics(y_prefixoids_inventory, n_prefixoids_inventory, 'Prefixoids')

        """

        n = len(dict_1.keys())

        y_candidates = dict_1.values()

        ind = np.arange(n)  # the x locations for the groups
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, y_candidates, width, color='y')

        n_candidates = dict_2.values()
        rects2 = ax.bar(ind + width, n_candidates, width, color='r')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Counts')
        ax.set_title('Counts per ' + title + ' candidate. Total: ' + str(sum(dict_1.values()) + sum(dict_2.values())) + '')
        ax.set_xticks(ind + width)
        ax.set_xticklabels((dict_1.keys()))

        ax.legend((rects1[0], rects2[0]), ('Y', 'N'))

        def autolabel(rects):
            # attach some text labels
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%d' % int(height),
                        ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        plt.show()

    # def search_duden_frequency(self, words_inventory):
    #     if type(words_inventory) != list:
    #         words_inventory = words_inventory.split()
    #
    #     def get_first_result(word):
    #         duden_url = 'http://www.duden.de/suchen/dudenonline/'
    #         r = requests.get(duden_url + word)
    #         data = r.text
    #         soup = BeautifulSoup(data, 'html.parser')
    #         try:
    #             main_sec = soup.find('section', id='block-duden-tiles-0')
    #             a_tags = [h2.a for h2 in main_sec.find_all('h2')]
    #             # print(a_tags[0].text)
    #             if a_tags[0].text == word:
    #                 return a_tags[0].get('href').split('/')[-1]
    #             else:
    #                 return 0
    #         except AttributeError:
    #             return 0
    #
    #     # needed for duden module
    #     def replace_umlauts(word_list):
    #         umlaute = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue', 'ß': 'ss'}
    #         if type(word_list) == list:
    #             new_list = []
    #             for word in word_list:
    #                 no_umlaut = word.translate({ord(k): v for k, v in umlaute.items()})
    #                 new_list.append(no_umlaut)
    #
    #             if len(word_list) == len(new_list):
    #                 return new_list
    #             else:
    #                 print('List error')
    #         if type(word_list) == str:
    #             return word_list.translate({ord(k): v for k, v in umlaute.items()})
    #         else:
    #             print('Replace Umlauts works only on strings and lists')
    #
    #     words_inventory = replace_umlauts(words_inventory)
    #     frequency_list = []
    #
    #     for w in words_inventory:
    #         words = duden.get(w)
    #         if words:
    #             try:
    #                 frequency_list.append(words.frequency)
    #             except AttributeError:
    #                 frequency_list.append(0)
    #         else:
    #             first_word = get_first_result(w)
    #             words = duden.get(first_word)
    #             try:
    #                 frequency_list.append(words.frequency)
    #             except AttributeError:
    #                 frequency_list.append(0)
    #
    #     return frequency_list


if __name__ == "__main__":
    """
        PREFIXOIDS
    """
    PREFIXOIDS = AffixoidClassifier('Prefixoids')
    prefixoid_inventory = PREFIXOIDS.read_file_to_list(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE)
    y_prefixoids_inventory = PREFIXOIDS.create_affixoid_inventory(DATA_FINAL_PATH+FINAL_PREFIXOID_FILE, 'Y')
    n_prefixoids_inventory = PREFIXOIDS.create_affixoid_inventory(DATA_FINAL_PATH+FINAL_PREFIXOID_FILE, 'N')
    print('Y:\t', y_prefixoids_inventory)
    print('N:\t', n_prefixoids_inventory)
    print('Total:\t', sum(y_prefixoids_inventory.values()) + sum(n_prefixoids_inventory.values()))
    print(len(prefixoid_inventory))
    # PREFIXOIDS.plot_statistics(y_prefixoids_inventory, n_prefixoids_inventory, 'Prefixoids')

    feature_1_prefixoids = []  # binary indicator, if affixoid
    feature_2_prefixoids = []  # frequency
    feature_3_prefixoids = []
    feature_3_prefixoids_lemmas = PREFIXOIDS.create_frequency_dictionary(DATA_PATH+FREQUENCY_PREFIXOID_LEMMAS)
    feature_3_prefixoids_heads = PREFIXOIDS.create_frequency_dictionary(DATA_PATH+FREQUENCY_PREFIXOID_HEADS)

    for i in prefixoid_inventory:
        item_1 = PREFIXOIDS.transform_to_binary(i[-1])
        item_2 = PREFIXOIDS.extract_frequency(i[-3], feature_3_prefixoids_lemmas)  # PREFIXOIDS.split_word_at_pipe(i[1])[0]
        item_3 = PREFIXOIDS.extract_frequency(PREFIXOIDS.split_word_at_pipe(i[1])[1], feature_3_prefixoids_heads)
        feature_1_prefixoids.append(item_1)
        feature_2_prefixoids.append(item_2)
        feature_3_prefixoids.append(item_3)
    # print(len(feature_1_prefixoids))
    # print(len(feature_2_prefixoids))
    # print(len(feature_3_prefixoids))

    # print(feature_1_prefixoids[50], feature_2_prefixoids[50], feature_3_prefixoids[50])

    """
        SUFFIXOIDS
    """
    SUFFIXOIDS = AffixoidClassifier('Suffixoids')
    suffixoid_inventory = SUFFIXOIDS.read_file_to_list(DATA_FINAL_PATH + FINAL_SUFFIXOID_FILE)
    y_suffixoids_inventory = SUFFIXOIDS.create_affixoid_inventory(DATA_FINAL_PATH+FINAL_SUFFIXOID_FILE, 'Y')
    n_suffixoids_inventory = SUFFIXOIDS.create_affixoid_inventory(DATA_FINAL_PATH+FINAL_SUFFIXOID_FILE, 'N')
    print('Y:\t', y_suffixoids_inventory)
    print('N:\t', n_suffixoids_inventory)
    print('Total:\t', sum(y_suffixoids_inventory.values()) + sum(n_suffixoids_inventory.values()))

    # print(sorted(y_suffixoids_inventory.items(), key=lambda kv: kv[0]))
    print(len(suffixoid_inventory))
    # SUFFIXOIDS.plot_statistics(y_suffixoids_inventory, n_suffixoids_inventory, 'Suffixoids')

    feature_1_suffixoids = []  # binary indicator, if affixoid
    feature_2_suffixoids = []  # frequency
    feature_3_suffixoids = []
    feature_3_suffixoids_lemmas = SUFFIXOIDS.create_frequency_dictionary(DATA_PATH + FREQUENCY_SUFFIXOID_LEMMAS)
    feature_3_suffixoids_modifiers = SUFFIXOIDS.create_frequency_dictionary(DATA_PATH + FREQUENCY_SUFFIXOID_MODIFIERS)

    for i in suffixoid_inventory:
        item_1 = SUFFIXOIDS.transform_to_binary(i[-1])
        item_2 = SUFFIXOIDS.extract_frequency(i[-3], feature_3_suffixoids_lemmas)
        item_3 = SUFFIXOIDS.extract_frequency(SUFFIXOIDS.split_word_at_pipe(i[1])[0], feature_3_suffixoids_modifiers)
        feature_1_suffixoids.append(item_1)
        feature_2_suffixoids.append(item_2)
        feature_3_suffixoids.append(item_3)
    # print(len(feature_1_suffixoids))
    # print(len(feature_2_suffixoids))
    # print(len(feature_3_suffixoids))

    # counter = 0
    # features = []
    # for i in prefixoid_inventory:
    #     f1 = feature_1_prefixoids[counter]
    #     f2 = feature_2_prefixoids[counter]
    #     f3 = feature_3_prefixoids[counter]
    #     f4 = i[-1]
    #     features.append([f1, f2, f3, f4])
    #     counter += 1
    # print(features)
    # print(feature_1_suffixoids[233], feature_2_suffixoids[233], feature_3_suffixoids[233])

