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
import re
import numpy as np
import matplotlib.pyplot as plt
# from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from pygermanet import load_germanet
ger = load_germanet()

################
# PATH SETTINGS
################
DATA_PATH = '../data/'
DATA_FINAL_PATH = '../data/final/'
DATA_FEATURES_PATH = '../data/features/'
DATA_RESSOURCES_PATH = '../res/'

################
# FILE SETTINGS
################
FINAL_PREFIXOID_FILE = 'binary_unique_instance_prefixoid_segmentations.txt'
FINAL_SUFFIXOID_FILE = 'binary_unique_instance_suffixoid_segmentations.txt'
FREQUENCY_PREFIXOID_FORMATIONS = 'lemma_frequencies_prefixoid_formations.csv'
FREQUENCY_SUFFIXOID_FORMATIONS = 'lemma_frequencies_suffixoid_formations.csv'
FREQUENCY_PREFIXOID_LEMMAS = 'prefixoid_lemmas_freqs.csv'
FREQUENCY_SUFFIXOID_LEMMAS = 'suffixoid_lemmas_freqs.csv'
FREQUENCY_PREFIXOID_HEADS = 'lemma_frequencies_unique_heads_of_prefixoid_formations.csv'
FREQUENCY_SUFFIXOID_MODIFIERS = 'modifiers_of_suffixoids_lemmas_freqs.csv'

FAST_TEXT_PREFIXOID_VECTORS = 'fastText/prefixoid-fastText-vectors.txt'
FAST_TEXT_SUFFIXOID_VECTORS = 'fastText/suffixoid-fastText-vectors.txt'


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
                >>> self.create_affixoid_inventory(DATA_FINAL_PATH+FINAL_PREFIXOID_FILE, 'Y')

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
                >>> self.read_file_to_list(DATA_FINAL_PATH+FINAL_PREFIXOID_FILE)

        """
        file_as_list = []
        with open(affixoid_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                if len(word) > 1:
                    file_as_list.append(word)
                else:
                    file_as_list.append(word[0])

        return file_as_list

    def write_list_to_file(self, affixoid_list, output_file):
        """ This function reads a file with affixoids and returns a list of lines from file

            Args:
                affixoid_list (list): List with affixoid instances
                output_file (file): Output file

            Returns:
                Output file

            Example:
                >>> self.write_list_to_file(['Bilderbuch'], 'out.txt')

        """
        f = open(output_file, 'w', encoding='utf-8')

        for item in affixoid_list:
            # ['Abfalldreck', 'Abfall|Dreck', 'Dreck', 'Schmutz', 'N']
            split_word = self.split_word_at_pipe(item[1])
            output_line = item[0] + '\t' + split_word[0] + '\t' + split_word[1]
            f.write(output_line + "\n")

        print('File written to: ', output_file)

        f.close()

    def transform_to_binary(self, class_name):
        """ This function transforms class labels to binary indicators

            Args:
                class_name (str): Class label

            Returns:
                Binary indicator for class label [0,1]

            Examples:
                >>> self.transform_to_binary('Y')

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
                >>> self.split_word_at_pipe('Bilderbuch|Absturz')

        """

        if '|' in word:
            return word.split('|')
        else:
            return [word, word]

    def extract_frequency(self, word, dictionary, return_as_vector=False):
        """ This function extracts frequencies for a given word from a dictionary

            Args:
                word (str): Word
                dictionary (dict): Dictionary with frequencies
                return_as_vector

            Returns:
                A frequency for a given word from a dictionary

            Examples:
                >>> self.extract_frequency('Bilderbuch', {'Bilderbuch':30})

        """
        if return_as_vector:
            return list(dictionary.values())

        if word in dictionary.keys():
            value = dictionary[word]
            return int(value)
        else:
            # TODO:
            # Ikone fehlt
            # Bhaishajya fehlt
            # Ostgoten fehlt
            # Splatterfilm fehlt
            return 1

    def create_frequency_dictionary(self, frequency_file):
        """ This function creates a dictionary with frequency instances of affixoids

            Args:
                frequency_file (file): File with affixoid frequencies

            Returns:
                Dictionary with frequency instances of affixoids

            Example:
                >>> self.create_frequency_dictionary(DATA_PATH+FREQUENCY_PREFIXOID_LEMMAS)

        """

        dictionary = {}
        with open(frequency_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                dict_key = word[0]
                if len(word) > 1:
                    dict_value = word[1]
                else:
                    dict_value = 0
                dictionary.update({dict_key: dict_value})

        return dictionary

    def create_vector_dictionary(self, vector_file):
        """ This function creates a dictionary with frequency instances of affixoids

            Args:
                frequency_file (file): File with affixoid frequencies

            Returns:
                Dictionary with frequency instances of affixoids

            Example:
                >>> self.create_vector_dictionary(DATA_PATH+FREQUENCY_PREFIXOID_LEMMAS)

        """

        dictionary = {}
        with open(vector_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                dict_key = word[0]
                dict_value = list(word[1:])
                dict_value_int = [float(x) for x in dict_value]
                dictionary.update({dict_key: dict_value_int})

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
                >>> self.plot_statistics(y_prefixoids_inventory, n_prefixoids_inventory, 'Prefixoids')

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

    def calculate_cosine_similarity(self, word_1, word_2, fast_text_vector_dict):
        """ This function ...

            Args:
                word_1 (string): 'Bilderbuchhochzeit'
                word_2 (string): 'Bilderbuch'
                fast_text_vector_dict (dict):

            Returns:
                Cosine similarity between word_1 and word_2

            Example:
                >>> self.calculate_cosine_similarity('Bilderbuchhochzeit', 'Bilderbuch', {'Bilderbuchhochzeit': [1], 'Bilderbuch': [1]}))

        """

        word_1_vec = fast_text_vector_dict[word_1]
        word_2_vec = fast_text_vector_dict[word_2]

        # print(word_1_vec)
        # print(word_2_vec)
        # print(word_1, word_2, cosine_similarity([word_1_vec], [word_2_vec])[0][0])
        # print("================")

        return cosine_similarity([word_1_vec], [word_2_vec])[0][0]

    def search_germanet(self, word, gn_supersenses_dict):
        """ This function ...

            Args:
                word (string): 'Bilderbuchhochzeit'
                gn_supersenses_dict (dict):

            Returns:
                Vector

            Example:
                >>> self.search_germanet()

        """

        gn_supersenses_dict_copy = gn_supersenses_dict.copy()

        regex = r"\(([^)]+)\)"
        r = r"(?<=\()(.*?)(?=\.)"
        word_synset = ger.synsets(word)
        print(word_synset)
        match_word = re.findall(regex, str(word_synset))

        if match_word:
            print(match_word, '>>>', word)
            for mt in match_word:
                m = ger.synset(mt).hypernym_paths
                print(m)
                for i in m:
                    # print(i)
                    for x in i:
                        # print(x)
                        ma = re.findall(r, str(x))
                        # print(ma)
                        if str(ma[0]) in gn_supersenses_dict.keys():
                            print(str(ma[0]), 1)
                            gn_supersenses_dict_copy.update({str(ma[0]): 1})
        else:
            pass

        print(list(gn_supersenses_dict_copy.values()))
        print('===================================')

        return list(gn_supersenses_dict_copy.values())


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
    germanet_supersenses_dict = {'Allgemein': 0, 'Bewegung': 0, 'Gefühl': 0, 'Geist': 0, 'Gesellschaft': 0, 'Körper': 0, 'Menge': 0, 'natürliches Phänomen': 0, 'Ort': 0, 'Pertonym Adjektiv': 0, 'Wahrnehmung': 0, 'privativspezifisch': 0, 'Relation': 0, 'Substanz': 0, 'Verhalten': 0, 'Zeit': 0, 'Artefakt': 0, 'Attribut': 0, 'Besitz': 0, 'Form': 0, 'Geschehen': 0, 'Gruppe': 0, 'Kognition': 0, 'Kommunikation': 0, 'Mensch': 0, 'Motiv': 0, 'Nahrung': 0, 'natürlicher Gegenstand': 0, 'Pflanze': 0, 'Tier': 0, 'Tops': 0, 'Körperfunktionsverb': 0, 'Konkurrenz': 0, 'Kontakt': 0, 'Lokation': 0, 'Kreatur': 0, 'Veränderung': 0, 'Verbrauch': 0}

    feature_1_prefixoids = []  # binary indicator, if affixoid
    feature_2_prefixoids = []  # frequency of complex word
    feature_3_prefixoids = []  # frequency of first part
    feature_4_prefixoids = []  # frequency of second part
    feature_5_prefixoids = []  # cosine similarity between complex word and head
    feature_6_prefixoids = []  # vector of GermaNet supersenses for complex word
    feature_7_prefixoids = []  # vector of GermaNet supersenses for first part
    feature_8_prefixoids = []  # vector of GermaNet supersenses for second part
    feature_3_prefixoids_formations = PREFIXOIDS.create_frequency_dictionary(DATA_PATH+FREQUENCY_PREFIXOID_FORMATIONS)
    feature_3_prefixoids_lemmas = PREFIXOIDS.create_frequency_dictionary(DATA_PATH+FREQUENCY_PREFIXOID_LEMMAS)
    feature_3_prefixoids_heads = PREFIXOIDS.create_frequency_dictionary(DATA_PATH+FREQUENCY_PREFIXOID_HEADS)
    feature_5_prefixoid_vector_dict = PREFIXOIDS.create_vector_dictionary(DATA_RESSOURCES_PATH + FAST_TEXT_PREFIXOID_VECTORS)

    for i in prefixoid_inventory:
        item_1 = PREFIXOIDS.transform_to_binary(i[-1])
        item_2 = PREFIXOIDS.extract_frequency(i[0], feature_3_prefixoids_formations)
        item_3 = PREFIXOIDS.extract_frequency(i[-3], feature_3_prefixoids_lemmas)  # PREFIXOIDS.split_word_at_pipe(i[1])[0]
        item_4 = PREFIXOIDS.extract_frequency(PREFIXOIDS.split_word_at_pipe(i[1])[1], feature_3_prefixoids_heads)
        item_5 = PREFIXOIDS.calculate_cosine_similarity(i[0], PREFIXOIDS.split_word_at_pipe(i[1])[0], feature_5_prefixoid_vector_dict)  # reverse for SUFFIXOIDS
        item_6 = PREFIXOIDS.search_germanet(i[0], germanet_supersenses_dict)
        item_7 = PREFIXOIDS.search_germanet(PREFIXOIDS.split_word_at_pipe(i[1])[0], germanet_supersenses_dict)
        item_8 = PREFIXOIDS.search_germanet(PREFIXOIDS.split_word_at_pipe(i[1])[1], germanet_supersenses_dict)
        feature_1_prefixoids.append(item_1)
        feature_2_prefixoids.append(item_2)
        feature_3_prefixoids.append(item_3)
        feature_4_prefixoids.append(item_4)
        feature_5_prefixoids.append(item_5)
        feature_6_prefixoids.append(item_6)
        feature_7_prefixoids.append(item_7)
        feature_8_prefixoids.append(item_8)
    # print(feature_1_prefixoids)
    # print(len(feature_1_prefixoids))
    # print(feature_2_prefixoids)
    # print(len(feature_2_prefixoids))
    # print(feature_3_prefixoids)
    # print(len(feature_3_prefixoids))
    # print(feature_4_prefixoids)
    # print(len(feature_4_prefixoids))
    # print(feature_5_prefixoids)
    # print(len(feature_5_prefixoids))
    # print(feature_6_prefixoids)
    # print(len(feature_6_prefixoids))
    # print(feature_7_prefixoids)
    # print(len(feature_7_prefixoids))
    # print(feature_8_prefixoids)
    # print(len(feature_8_prefixoids))
    zip_prefixoid_features = zip(prefixoid_inventory, feature_1_prefixoids, feature_2_prefixoids,
                        feature_3_prefixoids, feature_4_prefixoids, feature_5_prefixoids,
                        feature_6_prefixoids, feature_7_prefixoids, feature_8_prefixoids)

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
    feature_4_suffixoids = []
    feature_5_suffixoids = []
    feature_6_suffixoids = []
    feature_7_suffixoids = []
    feature_8_suffixoids = []
    feature_3_suffixoids_formations = PREFIXOIDS.create_frequency_dictionary(DATA_PATH + FREQUENCY_SUFFIXOID_FORMATIONS)
    feature_3_suffixoids_lemmas = SUFFIXOIDS.create_frequency_dictionary(DATA_PATH + FREQUENCY_SUFFIXOID_LEMMAS)
    feature_3_suffixoids_modifiers = SUFFIXOIDS.create_frequency_dictionary(DATA_PATH + FREQUENCY_SUFFIXOID_MODIFIERS)
    feature_5_suffixoids_vector_dict = SUFFIXOIDS.create_vector_dictionary(DATA_RESSOURCES_PATH + FAST_TEXT_SUFFIXOID_VECTORS)

    for i in suffixoid_inventory:
        item_1 = SUFFIXOIDS.transform_to_binary(i[-1])
        item_2 = SUFFIXOIDS.extract_frequency(i[0], feature_3_suffixoids_formations)
        item_3 = SUFFIXOIDS.extract_frequency(i[-3], feature_3_suffixoids_lemmas)
        item_4 = SUFFIXOIDS.extract_frequency(SUFFIXOIDS.split_word_at_pipe(i[1])[0], feature_3_suffixoids_modifiers)
        item_5 = SUFFIXOIDS.calculate_cosine_similarity(i[0], SUFFIXOIDS.split_word_at_pipe(i[1])[1], feature_5_suffixoids_vector_dict)
        item_6 = SUFFIXOIDS.search_germanet(i[0], germanet_supersenses_dict)
        item_7 = SUFFIXOIDS.search_germanet(SUFFIXOIDS.split_word_at_pipe(i[1])[0], germanet_supersenses_dict)
        item_8 = SUFFIXOIDS.search_germanet(SUFFIXOIDS.split_word_at_pipe(i[1])[1], germanet_supersenses_dict)
        feature_1_suffixoids.append(item_1)
        feature_2_suffixoids.append(item_2)
        feature_3_suffixoids.append(item_3)
        feature_4_suffixoids.append(item_4)
        feature_5_suffixoids.append(item_5)
        feature_6_suffixoids.append(item_6)
        feature_7_suffixoids.append(item_7)
        feature_8_suffixoids.append(item_8)
    # print(feature_1_suffixoids)
    # print(len(feature_1_suffixoids))
    # print(feature_2_suffixoids)
    # print(len(feature_2_suffixoids))
    # print(feature_3_suffixoids)
    # print(len(feature_3_suffixoids))
    # print(feature_4_suffixoids)
    # print(len(feature_4_suffixoids))
    # print(feature_5_suffixoids)
    # print(len(feature_5_suffixoids))
    # print(feature_6_suffixoids)
    # print(len(feature_6_suffixoids))
    # print(feature_7_suffixoids)
    # print(len(feature_7_suffixoids))
    # print(feature_8_suffixoids)
    # print(len(feature_8_suffixoids))
    zip_suffixoid_features = zip(suffixoid_inventory, feature_1_suffixoids, feature_2_suffixoids,
                                 feature_3_suffixoids, feature_4_suffixoids, feature_5_suffixoids,
                                 feature_6_suffixoids, feature_7_suffixoids, feature_8_suffixoids)

    # SUFFIXOIDS.write_list_to_file(suffixoid_inventory, 'suffix-out.txt')

    def test_case():
        print(germanet_supersenses_dict)
        print(PREFIXOIDS.transform_to_binary('Y'))
        print(PREFIXOIDS.extract_frequency('Bilderbuchabsturz', feature_3_prefixoids_formations))
        print(PREFIXOIDS.extract_frequency(PREFIXOIDS.split_word_at_pipe('Bilderbuch|Absturz')[0], feature_3_prefixoids_lemmas))
        print(PREFIXOIDS.extract_frequency(PREFIXOIDS.split_word_at_pipe('Bilderbuch|Absturz')[1], feature_3_prefixoids_heads))
        print(PREFIXOIDS.calculate_cosine_similarity('Bilderbuchabsturz', PREFIXOIDS.split_word_at_pipe('Bilderbuch|Absturz')[0], feature_5_prefixoid_vector_dict))
        print(PREFIXOIDS.search_germanet('Bilderbuchabsturz', germanet_supersenses_dict))
        print(PREFIXOIDS.search_germanet(PREFIXOIDS.split_word_at_pipe('Bilderbuch|Absturz')[0], germanet_supersenses_dict))
        print(PREFIXOIDS.search_germanet(PREFIXOIDS.split_word_at_pipe('Bilderbuch|Absturz')[1], germanet_supersenses_dict))

    # test_case()

    def write_features_to_file(instances_file, labels_file, zip_file):

        f = open(instances_file, 'w', encoding='utf-8')
        f2 = open(labels_file, 'w', encoding='utf-8')

        counter = 0

        for item in zip_file:

            i = [item[1], item[2], item[3], item[4], item[5]]

            for a in item[6]:
                i.append(a)

            for b in item[7]:
                i.append(b)

            for c in item[8]:
                i.append(c)

            print(i)

            f.write(str(i) + ',\n')
            f2.write(str(item[1]) + ',')
            counter += 1

        print('Instances written to: ', instances_file)
        print('Labels written to: ', labels_file)
        f.close()
        f2.close()

    write_features_to_file(DATA_FEATURES_PATH+'prefix-instances.txt', DATA_FEATURES_PATH+'prefix-labels.txt', zip_prefixoid_features)
    write_features_to_file(DATA_FEATURES_PATH+'suffix-instances.txt', DATA_FEATURES_PATH+'suffix-labels.txt', zip_suffixoid_features)
