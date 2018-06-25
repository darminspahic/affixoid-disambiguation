#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
affixoid_statistics

Short description:
This module writes various statistics about the affixoids_inventory.

License: MIT License
Version: 1.0

"""

# Import dependencies
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import duden
from pygermanet import load_germanet

DATA_FILES_PATH = '../../data/'
DATA_FILES_OUTPUT_PATH = '../../data/statistics/'


def affixoid_statistics(affixoid_inventory, data_file_path):
    """ This function iterates over txt files
        and writes various statistics about the affixes.

        Args:
            affixoid_inventory (str): Path to input file with affixoid inventory
            data_file_path (str): Path to input file with annotated data

        Returns:
            Written txt file for each file in the input path.

        Example:
            >>> affixoid_statistics(DATA_FILES_PATH + 'prefixoid_inventory.txt',
            DATA_FILES_PATH + 'binary_unique_instance_label_pairs_prefixoids.csv.affixoidal_status.tsv')

    """

    # Empty lists for collecting data
    affixoids_inventory = []
    affixoids_counter_list = []
    affixoids_y_counter_list = []
    affixoids_n_counter_list = []

    # print('Extracting affixoid statistics from:', data_file_path, 'to:', DATA_FILES_OUTPUT_PATH)

    with open(affixoid_inventory, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            affixoids_inventory.append(line)

    print('List of Affixoids: ', affixoids_inventory)
    print('Length of Affixoid candidates: ', len(affixoids_inventory))

    for a in affixoids_inventory:
        line_counter = 0
        y_counter = 0
        n_counter = 0
        with open(data_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split()
                if line[1].lower().startswith(a.lower()):
                    line_counter += 1
                    if len(line) < 4:
                        print('Missing data for: ', line)
                    if line[3] == 'Y':
                        y_counter += 1
                    if line[3] == 'N':
                        n_counter += 1
            affixoids_counter_list.append(line_counter)
            affixoids_y_counter_list.append(y_counter)
            affixoids_n_counter_list.append(n_counter)

    with open(data_file_path, 'r', encoding='utf-8') as f:
        line_counter = 0

        for line in f:
            line_counter += 1

    print('Data length: ', line_counter)
    print('Data per candidate: ', affixoids_counter_list)
    print('Y candidate: ', affixoids_y_counter_list)
    print('N candidate: ', affixoids_n_counter_list)
    print(sum(affixoids_counter_list))

    return [affixoids_inventory, affixoids_counter_list, affixoids_y_counter_list, affixoids_n_counter_list]


def affixoid_extractor(affixoid_inventory):
    """ This function

        Args:


        Returns:


        Example:
            >>> affixoid_extractor(DATA_FILES_PATH + 'unique_binary_prefixoid_formations.srt')

    """

    # Empty lists for collecting data
    affixoids_inventory = []

    with open(affixoid_inventory, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            affixoids_inventory.append(word)

    return affixoids_inventory


def search_duden(words_inventory):
    found_array = []
    not_found_array = []

    for w in words_inventory:
        word = duden.get(w)
        if word:
            try:
                print(word.name)
            except:
                print(word)
            found_array.append(w)
        else:
            not_found_array.append(w)

    print('Found: ', found_array)
    print('Total found: ', len(found_array))
    print('--------------------')
    print('Not found: ', not_found_array)
    print('Total not found: ', len(not_found_array))


def search_germanet(words_inventory):
    found_array = []
    not_found_array = []

    gn = load_germanet()

    for t in words_inventory:
        if len(gn.synsets(t)) > 0:
            print(gn.synsets(t))
            found_array.append(t)
        else:
            not_found_array.append(t)

    print('Found: ', found_array)
    print('Total found: ', len(found_array))
    print('--------------------')
    print('Not found: ', not_found_array)
    print('Total not found: ', len(not_found_array))


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


def plot_statistics(arguments, title):
    """ This function plots charts with affixoid statistics.

        Args:
            arguments (list): Returned list from affixoid_statistics()
            title (str): Title of the chart

        Returns:
            Matplotlib images

        Example:
            >>> plot_statistics(arguments_1, 'prefixoid')

    """

    n = len(arguments[0])
    y_candidates = arguments[2]

    ind = np.arange(n)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, y_candidates, width, color='y')

    n_candidates = arguments[3]
    rects2 = ax.bar(ind + width, n_candidates, width, color='r')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Counts')
    ax.set_title('Counts per '+title+' candidate. Total: '+str(sum(arguments[1]))+'')
    ax.set_xticks(ind + width)
    ax.set_xticklabels((arguments[0]))

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


# Disable print
def disable_print():
    """ This function disables printing.

        Args:

        Returns:
            none

        Example:
            >>> disable_print()

    """
    sys.stdout = open(os.devnull, 'w')


# Enable print
def enable_print():
    """ This function enables printing.

            Args:

            Returns:
                none

            Example:
                >>> enable_print()

        """
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    # disable_print()
    print('Extracting prefixoids:')
    arguments_1 = affixoid_statistics(DATA_FILES_PATH + 'prefixoid_inventory.txt', DATA_FILES_PATH + 'binary_unique_instance_label_pairs_prefixoids.csv.affixoidal_status.tsv')
    plot_statistics(arguments_1, 'prefixoid')

    print('--------------------')

    print('Extracting suffixoids:')
    arguments_2 = affixoid_statistics(DATA_FILES_PATH + 'suffixoid_inventory.txt', DATA_FILES_PATH + 'binary_unique_instance_label_pairs_suffixoids.csv.affixoidal_status.tsv')
    plot_statistics(arguments_2, 'suffixoid')

    # test_array = ['Löffel', 'Ballspiel', 'Anschlagbolzen', 'Anschlußbolzen', 'Anstandspapst', 'ginge']
    # new_test_array = replace_umlauts(test_array)
    # print(new_test_array)
    #
    # search_duden_frequency(test_array)
    # search_duden_frequency(new_test_array)
    # print('--------------------')

    print('Extracting prefixoids:')
    prefixoids = affixoid_extractor(DATA_FILES_PATH + 'unique_binary_prefixoid_formations.srt')
    prefixoids_no_umlauts = replace_umlauts(prefixoids)
    # search_duden_frequency(prefixoids_no_umlauts)
    search_germanet(prefixoids)
    print('--------------------')

    print('Extracting suffixoids:')
    suffixoids = affixoid_extractor(DATA_FILES_PATH + 'unique_binary_suffixoid_formations.srt')
    suffixoids_no_umlauts = replace_umlauts(suffixoids)
    # search_duden_frequency(suffixoids_no_umlauts)
    search_germanet(suffixoids)

