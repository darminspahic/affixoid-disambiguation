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
from pygermanet import load_germanet

DATA_FILES_PATH = '../../data/'
DATA_FILES_PATH_FINAL = '../../data/final/'
DATA_FILES_OUTPUT_PATH = '../../data/statistics/'


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


if __name__ == "__main__":
    y_prefixoids_inventory = create_affixoid_inventory(DATA_FILES_PATH_FINAL + 'binary_unique_instance_prefixoid_segmentations.txt', 'Y')
    n_prefixoids_inventory = create_affixoid_inventory(DATA_FILES_PATH_FINAL + 'binary_unique_instance_prefixoid_segmentations.txt', 'N')
    # print('Y prefixoids:', y_prefixoids_inventory)
    # print('N prefixoids:', n_prefixoids_inventory)

    y_suffixoids_inventory = create_affixoid_inventory(DATA_FILES_PATH_FINAL + 'binary_unique_instance_suffixoid_segmentations.txt', 'Y')
    n_suffixoids_inventory = create_affixoid_inventory(DATA_FILES_PATH_FINAL + 'binary_unique_instance_suffixoid_segmentations.txt', 'N')
    # print('Y suffixoids:', y_suffixoids_inventory)
    # print('N suffixoids:', n_suffixoids_inventory)
    # print(sorted(y_suffixoids_inventory.items(), key=lambda kv: kv[0]))

    prefixoid_feature_inventory = []

    # with open(DATA_FILES_PATH_FINAL + 'binary_unique_instance_prefixoid_segmentations.txt', 'r', encoding='utf-8') as f:
    #     for line in f:
    #         word = line.strip().split()
    #         f_0 = word[0]  # keep full word for easier controlling; not a feature
    #         f_1 = transform_to_binary(word[-1])  # FEATURE 1: binary indicator if affixoid
    #         f_2 = search_duden_frequency(word[0])  # FEATURE 2: Duden Frequency of complex word; search_duden_frequency(word[1])
    #         f_3 = search_duden_frequency(split_word_at_pipe(word[1]))  # FEATURE 3: Duden frequency of both words; search_duden_frequency(split_word_at_pipe(word[1]))
    #         f_4 = get_dictionary_frequency(y_prefixoids_inventory, n_prefixoids_inventory, word[-3], word[-1])  # FEATURE 4: Word frequency
    #         f_5 = word[-1]  # keep class name for easier controlling
    #         prefixoid_feature_inventory.append([f_0, f_1, f_2, f_3, f_4, f_5])

    # print(prefixoid_feature_inventory)
    # print(search_duden_frequency('Traumstele'))
    # print(search_duden_frequency(split_word_at_pipe('Traum|stele')))
    # print(sum(y_prefixoids_inventory.values()) + sum(n_prefixoids_inventory.values()))
    # print(sum(y_suffixoids_inventory.values()) + sum(n_suffixoids_inventory.values()))

    # print(get_dictionary_frequency(y_prefixoids_inventory, n_prefixoids_inventory, 'Jahrhundert', 'Y'))
    # print(get_dictionary_frequency(y_suffixoids_inventory, n_suffixoids_inventory, 'König', 'N'))

    # Y candidate: 999
    # N candidate: 1010

    # 2009
    # 1873

    # ycandidate = [138, 123, 71, 86, 50, 156, 82, 17, 147, 129]
    # ncandidate = [63, 78, 130, 115, 151, 45, 119, 184, 54, 71]
    #
    # y_dictionary = dict(zip(fixoids, ycandidate))
    # print(y_dictionary)
    # print(sum(y_dictionary.values()))
    #
    # n_dictionary = dict(zip(fixoids, ncandidate))
    # print(n_dictionary)
    # print(sum(n_dictionary.values()))
    #
    # total = sum(n_dictionary.values()) + sum(y_dictionary.values())
    #
    # print(total)
    #
    # print(total / n_dictionary['Bilderbuch'])
    # print(total / y_dictionary['Bilderbuch'])

    # print(search_duden_frequency(['Bilderbuchbabe', 'besitzen', 'Katze', 'ausleihen', 'Alter', 'Bilderbuchkarriere', 'Bilderbuchabsturz']))
    # print(search_duden_frequency('Bilderbuchbabe'))
    # print(search_duden_frequency('Alter'))
    # print(search_duden_frequency('Babe'))
    # print(search_duden_frequency(split_word_at_pipe('Traum|Tänzer')))
    # print(search_duden_frequency(split_word_at_pipe('Bilderbuch|Babe')))
    # #
    # # get_first_result('Alter')
    # get_first_result('Babe')

    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    Traumstadion = [1.51038, 0.3298, -0.2268, 0.20675, 0.38643, 0.59705, -0.087452, 0.59164, 0.43811, 0.5675, -0.11306,
                    0.38801, -0.29795, -0.12553, 0.25619, 0.55814, -0.27238, 0.69037, 0.33325, -0.61344, 0.0051162,
                    0.086551, 0.098419, 0.55631, -0.30947, -0.64671, 0.31984, 0.20901, 0.42472, -0.408, -0.55522,
                    0.28909, -0.33714, -0.96021, 0.37335, -0.2752, -0.39406, -0.31114, -0.23886, -0.32193, 0.82913,
                    -0.301, -0.23072, 0.2955, 0.12953, -1.2596, 0.1847, 0.011893, 0.029221, -1.8696, 0.46018, -0.077334,
                    0.31618, -0.23598, -0.33036, -0.20551, -0.21024, -0.66147, 0.12498, 1.1762, -0.46383, -0.26002,
                    0.84079, 0.73742, -0.55072, -0.41187, -0.78672, 0.068199, 0.54515, 0.76482, -0.063293, 0.92447,
                    -0.39307, -0.92211, -0.0068755, -0.43408, 0.31891, -0.19941, -0.13035, -0.51669, -0.71989, -0.18618,
                    0.5848, 0.20711, -0.35254, -0.25175, -0.051734, 0.25907, -0.22623, 0.08681, 0.57146, -0.064433,
                    -0.20164, 0.083979, 0.44955, -0.069446, 0.61753, -0.027496, 0.042363, -0.39738]

    Traum = [0.0046503, 0.18989, -0.42962, -0.23761, -0.37157, 0.2073, 0.26476, 0.37123, 0.44159, 0.94143, -0.021516,
             0.24184, -0.072423, -0.10924, 0.18458, 0.077331, -0.12824, -0.11693, -0.0016658, 0.28182, 0.2657, 0.46063,
             -0.20145, 0.17755, -0.10544, 0.39572, -0.2781, 0.1048, -0.45059, -0.39746, -0.048122, 0.49525, 0.12317,
             -0.7438, 0.41821, 0.12727, -0.040206, -0.31718, -0.61842, 0.20747, 0.11066, 0.29706, -0.6208, 0.0067448,
             -0.21254, -0.77991, 0.051589, -0.17453, -0.02899, -0.62576, 0.098076, 0.074, -0.57697, 0.3432, 0.26346,
             -0.17234, -0.47017, -0.11664, 0.23176, 0.23599, -0.22813, -0.19857, 0.45183, 0.15496, -0.70392, -0.52697,
             -1.1811, -0.074139, 0.30198, 0.50106, -0.11858, 0.462, -0.31422, -0.46156, 0.43266, 0.018653, -0.21436,
             0.47408, -0.12549, -0.017468, -0.25579, 0.25312, 0.43324, 0.18564, 0.019271, 0.010571, 0.45678, 0.14738,
             -0.22385, 0.15429, 0.32365, 0.05601, 0.30289, -0.14185, -0.14232, 0.12013, 0.28125, -0.048597, 0.30315,
             -0.82039]

    Stadion = [-0.41614, -0.19398, -0.094264, 0.1172, 0.17098, 0.1482, -0.15405, 0.58524, 0.21249, 0.55379, -0.076702,
               -0.085688, -0.11081, 0.056151, 0.29642, 0.37313, -0.23515, 0.59835, 0.58247, -0.23424, 0.027191,
               0.096253, 0.26035, 0.40325, -0.49294, -0.51477, 0.26817, -0.035959, 0.33958, 0.024145, -0.53692,
               0.075231, -0.47967, -0.49972, -0.052359, -0.30366, -0.31441, -0.24431, -0.34808, -0.20592, 1.1228,
               -0.36317, -0.34896, -0.17516, 0.11645, -1.3344, -0.16618, 0.18677, 0.017778, -1.479, 0.69317, -0.047809,
               0.29539, -0.044451, -0.34304, 0.041009, 0.064336, -0.68135, 0.18749, 0.88533, -0.31594, 0.15115, 0.51806,
               0.2513, -0.24916, -0.17946, -0.30837, 0.29312, 0.043719, 0.45863, -0.15788, 0.64206, 0.014545, -0.50693,
               -0.093965, -0.42731, 0.36565, -0.64494, 0.031658, -0.58889, -0.8822, -0.23092, 0.32911, 0.23285,
               -0.39345, -0.44759, 0.072123, 0.16015, -0.31945, 0.22186, 0.2083, -0.13704, -0.45956, 0.16202, 0.52608,
               0.18826, 0.82798, -0.012511, -0.74997, -0.090543]

    print(cosine_similarity([Traumstadion], [Traum]))
    print(cosine_similarity([Traumstadion], [Stadion]))
    print(np.sum([Traum, Stadion]))



