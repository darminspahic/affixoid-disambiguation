#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
affixoid_extractor

Short description:
This module writes various statistics about the affixoids_inventory.

License: MIT License
Version: 1.0

"""
import duden
from pygermanet import load_germanet

DATA_FILES_PATH = '../../data/'
DATA_FILES_OUTPUT_PATH = '../../data/statistics/'


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
    new_list = []
    for word in word_list:
        no_umlaut = word.translate({ord(k): v for k, v in umlaute.items()})
        new_list.append(no_umlaut)

    if len(word_list) == len(new_list):
        return new_list
    else:
        print('List error')



if __name__ == "__main__":
    # test_array = ['Löffel', 'Ballspiel', 'Anschlagbolzen', 'Anschlußbolzen', 'Anstandspapst', 'ginge']
    # new_test_array = replace_umlauts(test_array)
    # print(new_test_array)
    #
    # search_duden(test_array)
    # search_duden(new_test_array)
    # print('--------------------')

    # print('Extracting prefixoids:')
    # prefixoids = affixoid_extractor(DATA_FILES_PATH + 'unique_binary_prefixoid_formations.srt')
    # prefixoids_no_umlauts = replace_umlauts(prefixoids)
    # search_duden(prefixoids_no_umlauts)
    # search_germanet(prefixoids)
    # print('--------------------')

    print('Extracting suffixoids:')
    suffixoids = affixoid_extractor(DATA_FILES_PATH + 'unique_binary_suffixoid_formations.srt')
    # suffixoids_no_umlauts = replace_umlauts(suffixoids)
    # search_duden(suffixoids_no_umlauts)
    search_germanet(suffixoids)

