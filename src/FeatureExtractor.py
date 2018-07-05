#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
FeatureExtractor

Short description:
TODO
This module...

License: MIT License
Version: 1.0

"""

import sys
import os
# import duden
# import requests
import re
import numpy as np
import matplotlib.pyplot as plt
import ast
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
# from pygermanet import load_germanet
# ger = load_germanet()

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

SENTIMERGE_POLARITY = 'SentiMerge/sentimerge.txt'

################
# GermaNet
################
GERMANET_XML_PATH = 'modules/GN_V120_XML/'

"""GermaNet Supersenses"""
GN_SUPERSENSES = {'Allgemein': 0, 'Bewegung': 0, 'Gefuehl': 0, 'Geist': 0, 'Gesellschaft': 0, 'Koerper': 0, 'Menge': 0, 'natPhaenomen': 0, 'Ort': 0, 'Pertonym': 0, 'Perzeption': 0, 'privativ': 0, 'Relation': 0, 'Substanz': 0, 'Verhalten': 0, 'Zeit': 0, 'Artefakt': 0, 'Attribut': 0, 'Besitz': 0, 'Form': 0, 'Geschehen': 0, 'Gruppe': 0, 'Kognition': 0, 'Kommunikation': 0, 'Mensch': 0, 'Motiv': 0, 'Nahrung': 0, 'natGegenstand': 0, 'Pflanze': 0, 'Tier': 0, 'Tops': 0, 'Koerperfunktion': 0, 'Konkurrenz': 0, 'Kontakt': 0, 'Lokation': 0, 'Schoepfung': 0, 'Veraenderung': 0, 'Verbrauch': 0}

"""To prevent unnecessary parsing, use words already found"""
GN_PREF_FORMATIONS = ['Blitzgerät', 'Blitzkarriere', 'Blitzkrieg', 'Blitzkurs', 'Blitzlampe', 'Blitzröhre', 'Blitzschach', 'Blitzschlag', 'Bombenabwurf', 'Bombenalarm', 'Bombendrohung', 'Bombenexplosion', 'Bombenleger', 'Bombennacht', 'Bombenopfer', 'Bombenschacht', 'Bombenschaden', 'Bombensplitter', 'Bombenteppich', 'Bombentest', 'Bombentrichter', 'Glanzente', 'Glanzleistung', 'Glanzlicht', 'Glanzpunkt', 'Glanzrolle', 'Glanzstoff', 'Glanzstück', 'Glanzzeit', 'Jahrhundertfeier', 'Jahrhunderthälfte', 'Jahrhunderthochwasser', 'Jahrhundertsommer', 'Jahrhundertwechsel', 'Qualitätsbewusstsein', 'Qualitätskriterium', 'Qualitätsprüfung', 'Qualitätsstandard', 'Qualitätsverbesserung', 'Qualitätswein', 'Qualitätszuwachs', 'Schweineblut', 'Schweinebraten', 'Schweinefleisch', 'Schweinehaltung', 'Schweinehirte', 'Schweinepest', 'Spitzenfunktionär', 'Spitzenkandidatin', 'Spitzenkoch', 'Spitzenläufer', 'Spitzenleistung', 'Spitzenplatz', 'Spitzenreiter', 'Spitzenspiel', 'Spitzensportler', 'Spitzenverband', 'Spitzenverdiener', 'Spitzenverein', 'Spitzenwert', 'Traummädchen', 'Traumwelt']
GN_SUFF_FORMATIONS = ['Börsenguru', 'Burgunderkönig', 'Bürohengst', 'Dänenkönig', 'Donnergott', 'Dreikönig', 'Feenkönig', 'Feuergott', 'Froschkönig', 'Gegenkönig', 'Gegenpapst', 'Gotenkönig', 'Gottkönig', 'Großkönig', 'Hausschwein', 'Herzkönig', 'Himmelsgott', 'Hochkönig', 'Hunnenkönig', 'Kleinkönig', 'Langobardenkönig', 'Liebesgott', 'Märchenkönig', 'Marienikone', 'Meeresgott', 'Moralapostel', 'Normannenkönig', 'Perserkönig', 'Preußenkönig', 'Priesterkönig', 'Rattenkönig', 'Schlagbolzen', 'Schöpfergott', 'Schützenkönig', 'Schwedenkönig', 'Slawenapostel', 'Sonnenkönig', 'Stammapostel', 'Torschützenkönig', 'Unterkönig', 'Vizekönig', 'Vogelkönig', 'Wachtelkönig', 'Warzenschwein', 'Westgotenkönig', 'Wettergott', 'Wildschwein', 'Winterkönig', 'Zaunkönig', 'Zuchthengst', 'Zwergenkönig']


class FeatureExtractor:
    """ This is the main module and it is a collection of all
        modules from the project.

        Returns: Files with feature vectors

        Example: PREF = FeatureExtractor()

    """

    def __init__(self, string):
        print('=' * 40)
        print("Running FeatureExtractor on:", string)
        print('-' * 40)

    def create_affixoid_dictionary(self, affixoid_file, class_name):
        """ This function creates a dictionary with class instances of affixoids

            Args:
                affixoid_file (file): File with affixoid instances
                class_name (str): Class label (Y|N)

            Returns:
                Dictionary with class instances

            Example:
                >>> self.create_affixoid_dictionary(DATA_FINAL_PATH+FINAL_PREFIXOID_FILE, 'Y')

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
        """ This function reads a tab-separated file with affixoids and returns a list of lines from file

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

    def write_list_to_file(self, input_list, output_file, item_range=-1, split_second_word=False):
        """ This function reads a list with affixoids or features and writes lines to a file

            Args:
                input_list (list): List with affixoid instances or features
                output_file (file): Output file
                item_range (int): indicator to which index the list returns a line
                split_second_word (bool): Split second word in lists ['Abfalldreck', 'Abfall|Dreck', 'Dreck', 'Schmutz', 'N']

            Returns:
                Output file

            Example:
                >>> self.write_list_to_file(['Bilderbuch', 'Absturz'], 'out.txt')

        """
        f = open(output_file, 'w', encoding='utf-8')

        for item in input_list:
            if split_second_word:
                split_word = self.split_word_at_pipe(item[1])
                output_line = item[0] + '\t' + split_word[0] + '\t' + split_word[1]

            else:
                if type(item) == list:
                    sublist = item
                    output_line = '\t'.join([str(x) for x in sublist])

                else:
                    output_line = item

            f.write(str(output_line) + '\n')

        print('File written to: ', output_file)

        f.close()

    def transform_class_name_to_binary(self, class_name):
        """ This function transforms class labels to binary indicators

            Args:
                class_name (str): Class label (Y|N)

            Returns:
                Binary indicator for class label [0,1]

            Example:
                >>> self.transform_class_name_to_binary('Y')

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

    def extract_frequency(self, word, dictionary, return_as_binary_vector=False):
        """ This function extracts frequencies for a given word from a dictionary of frequencies

            Args:
                word (str): Word
                dictionary (dict): Dictionary with frequencies
                return_as_binary_vector (bool): Returns full vector with binary indicator where the word is found

            Returns:
                A frequency for a given word from a dictionary

            Examples:
                >>> self.extract_frequency('Bilderbuch', {'Bilderbuch':30})

        """
        if return_as_binary_vector:
            dictionary_copy = dictionary.fromkeys(dictionary, 0)
            dictionary_copy.update({word: 1})
            return list(dictionary_copy.values())

        if word in dictionary.keys():
            value = dictionary[word]
            return int(value)

        else:
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
        """ This function creates a dictionary with vector values from affixoids

            Args:
                vector_file (file): File with vector values from FastText

            Returns:
                Dictionary with vector values as list

            Example:
                >>> self.create_vector_dictionary(DATA_PATH+FREQUENCY_PREFIXOID_LEMMAS)

        """

        dictionary = {}
        with open(vector_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                dict_key = word[0]
                dict_value = list(word[1:])
                dict_value_float = [float(x) for x in dict_value]
                dictionary.update({dict_key: dict_value_float})

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
                >>> self.plot_statistics(y_pref_dict, n_pref_dict, 'Prefixoids')

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
        """ This function calculates cosine similarity between two words, using vector data from a FastText model

            Args:
                word_1 (string): 'Bilderbuchhochzeit'
                word_2 (string): 'Bilderbuch'
                fast_text_vector_dict (dict): Vector data from a FastText model

            Returns:
                Cosine similarity between word_1 and word_2

            Example:
                >>> self.calculate_cosine_similarity('Bilderbuchhochzeit', 'Bilderbuch', {'Bilderbuchhochzeit': [1], 'Bilderbuch': [1]}))

        """

        try:
            word_1_vec = fast_text_vector_dict[word_1]
            word_2_vec = fast_text_vector_dict[word_2]
            return cosine_similarity([word_1_vec], [word_2_vec])[0][0]

        except KeyError:
            print('Words not found in fastText vector dictionary')
            return 0

    def parse_germanet(self, germanet_directory):
        """TODO"""

        xml_content = []

        for root, dirnames, filenames in os.walk(germanet_directory):
            for filename in filenames:
                if (filename.startswith('adj') or filename.startswith('nomen') or filename.startswith('verben')) and filename.endswith('.xml'):
                    fname = os.path.join(root, filename)
                    print('Reading file into memory: {}'.format(fname))
                    with open(fname, 'r', encoding='utf-8') as input_xml:
                        print('Parsing xml: {}'.format(fname))
                        soup = BeautifulSoup(input_xml, 'lxml-xml')
                        xml_content.append(soup)

        print('Files from {}'.format(germanet_directory), 'ready to parse.')

        return xml_content

    def search_germanet(self, word, gn_supersenses_dict, xml_soup, return_single_word=False):
        """ This function searches through GermaNet for supersenses noted in germanet_supersenses_dict
            and returns a vector with binary indicator where the sense has been found.

            Args:
                word (string): 'Bilderbuchhochzeit'
                gn_supersenses_dict (dict): Dictionary with GermaNet supersenses
                xml_soup (list): Parsed xml files with Beautiful soup
                return_single_word (bool): Return single word when set to true

            Returns:
                Vector with length of germanet_supersenses_dict.values()

            Example:
                >>> self.search_germanet('Husky', {'Tier': 0, 'Mensch': 0})

        """

        print('Searching for: ', word)

        found_word = ''
        found_supersense = ''

        gn_supersenses_dict_copy = gn_supersenses_dict.copy()

        if return_single_word:
            for item in xml_soup:
                all_synsets = item.find_all('synset')
                for syn in all_synsets:
                    supersense = syn.get('class')
                    all_forms = syn.find_all('orthForm')
                    # all_modifiers = syn.find_all('modifier')
                    # all_heads = syn.find_all('head')
                    for form in all_forms:
                        if form.get_text() == word:
                            found_word = word
                            found_supersense = supersense
                            gn_supersenses_dict_copy.update({supersense: 1})

            if len(found_supersense) > 0:
                print('Word found:', found_word)
                return found_word
            else:
                print('Word:', word, 'not found in GermaNet')
                return False

        else:
            for item in xml_soup:
                all_synsets = item.find_all('synset')
                for syn in all_synsets:
                    supersense = syn.get('class')
                    all_forms = syn.find_all('orthForm')
                    # all_modifiers = syn.find_all('modifier')
                    # all_heads = syn.find_all('head')
                    for form in all_forms:
                        if form.get_text() == word:
                            found_word = word
                            found_supersense = supersense
                            gn_supersenses_dict_copy.update({supersense: 1})

            if len(found_supersense) > 0:
                print(word, 'supersense >', found_supersense)
            else:
                print('Supersense for:', word, 'not found in GermaNet')

        return list(gn_supersenses_dict_copy.values())

    def return_similar_cosine_word(self, word, splitword, fast_text_vector_dict, from_germanet=True, polarity_dict=None, threshold=0.7):
        """ This function calculates cosine similarity between the input word, its corresponding splitword
            and all other words from fast_text_vector_dict. It returns the most similar word based on cosine similarity.

            Args:
                word (string): 'Bilderbuchhochzeit'
                splitword (string): 'Bilderbuch|Hochzeit'
                fast_text_vector_dict (dict): Vector data from a FastText model
                from_germanet (bool): Return the word if it is found in GermaNet
                polarity_dict dict(): Dictionary with polarities
                threshold (float): Indicator which cosine values to consider

            Returns:
                Most similar word

            Example:
                >>> self.return_similar_cosine_word('Bilderbuchhochzeit', 'Bilderbuch|Hochzeit', fast_text_vector_dict, threshold=0.7, polarity_dict={})

        """

        print('Searching for similar word to: ', word)

        word_1 = word
        word_2 = self.split_word_at_pipe(splitword)[0]
        word_3 = self.split_word_at_pipe(splitword)[1]
        most_similar_word = ''

        cosine_similarity_dict = {}
        for key in fast_text_vector_dict.keys():
            # TODO: Check if word_1 != key and word_2 != key and word_3 != key:
            if word_1 != key and word_2 != key and word_3 != key:
                cs_1 = self.calculate_cosine_similarity(word_1, key, fast_text_vector_dict)
                cs_2 = self.calculate_cosine_similarity(word_2, key, fast_text_vector_dict)
                cs_3 = self.calculate_cosine_similarity(word_3, key, fast_text_vector_dict)
                if cs_1 > threshold:
                    cosine_similarity_dict.update({str(key): cs_1})
                if cs_2 > threshold:
                    cosine_similarity_dict.update({str(key): cs_2})
                if cs_3 > threshold:
                    cosine_similarity_dict.update({str(key): cs_3})
        # print(cosine_similarity_dict)

        sorted_cosine_similarity_dict = sorted(cosine_similarity_dict.items(), key=lambda kv: kv[1], reverse=True)
        # print(len(sorted_cosine_similarity_dict))

        if from_germanet:
            print('Searching in GermaNet')
            for x in sorted_cosine_similarity_dict:
                if word_1 not in GN_PREF_FORMATIONS or i not in GN_SUFF_FORMATIONS:
                    print('Word not in GN dicts. Continuing...')
                    continue
                else:
                    print('Similar word in fastText dictionary: ', x[0])
                    print('Cosine similarity: ', x[1])
                    # TODO: Fix this
                    gn_word = self.search_germanet(x[0], gn_supersenses_dict=GN_SUPERSENSES, xml_soup=germanet_xml, return_single_word=True)
                    if gn_word:
                        print(gn_word)
                        most_similar_word = x[0]
                        #break
                        return most_similar_word
        else:
            print('Searching in polarity_dict')
            for x in sorted_cosine_similarity_dict:
                print('Similar word found: ', x[0])
                print('Cosine similarity: ', x[1])
                if x[0] in polarity_dict.keys():
                    print('Most similar word from polarity dictionary:', x[0])
                    most_similar_word = x[0]
                    # break
                    return most_similar_word

    def create_polarity_dict(self, polarity_file):
        """TODO"""

        dictionary = {}

        with open(polarity_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                # dict_key = word[0]
                if word[1] == 'N' or word[1] == 'NE':
                    dict_key = word[0].capitalize()
                else:
                    dict_key = word[0]
                dict_value = word[2]
                dictionary.update({dict_key: dict_value})

        return dictionary

    def extract_polarity(self, word, polarity_dict):
        """TODO Docstrings"""
        # try:
        #     value = polarity_dict[word]
        #     v = ast.literal_eval(value)
        # except KeyError:
        #     try:
        #         value = polarity_dict[word.lower()]
        #         v = ast.literal_eval(value)
        #     except KeyError:
        #         value = 0
        #         v = 0
        # # v = ast.literal_eval(value)
        # return v

        if word in polarity_dict.keys():
            value = polarity_dict[word]
            v = ast.literal_eval(value)
            return v

        else:
            return 0


class Style:
    BOLD = '\033[1m'
    END = '\033[0m'


if __name__ == "__main__":
    """
        PREFIXOIDS
    """
    PREF = FeatureExtractor('Prefixoids')
    pref_inventory_list = PREF.read_file_to_list(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE)
    suff_inventory_list = PREF.read_file_to_list(DATA_FINAL_PATH + FINAL_SUFFIXOID_FILE)
    y_pref_dict = PREF.create_affixoid_dictionary(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE, 'Y')
    n_pref_dict = PREF.create_affixoid_dictionary(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE, 'N')
    print('Y:\t', y_pref_dict)
    print('N:\t', n_pref_dict)
    print('Total:\t', sum(y_pref_dict.values()) + sum(n_pref_dict.values()))
    print(len(pref_inventory_list))
    # PREF.plot_statistics(y_pref_dict, n_pref_dict, 'Prefixoids')
    germanet_xml = PREF.parse_germanet(GERMANET_XML_PATH)

    f0_pref_list = []  # prefix coded into dictionary
    f1_pref_list = []  # binary indicator, if affixoid
    f2_pref_list = []  # frequency of complex word
    f3_pref_list = []  # frequency of first part
    f4_pref_list = []  # frequency of second part
    f5_pref_list = []  # cosine similarity between complex word and head
    f6_pref_list = []  # vector of GermaNet supersenses for complex word
    f7_pref_list = []  # vector of GermaNet supersenses for first part
    f8_pref_list = []  # vector of GermaNet supersenses for second part
    f9_pref_list = []  # SentiMerge Polarity for complex word
    f10_pref_list = []  # SentiMerge Polarity for first part
    f11_pref_list = []  # SentiMerge Polarity for second part

    f2_pref_formations = PREF.create_frequency_dictionary(DATA_PATH + FREQUENCY_PREFIXOID_FORMATIONS)
    f3_pref_lemmas = PREF.create_frequency_dictionary(DATA_PATH + FREQUENCY_PREFIXOID_LEMMAS)
    f4_pref_heads = PREF.create_frequency_dictionary(DATA_PATH + FREQUENCY_PREFIXOID_HEADS)
    f5_pref_vector_dict = PREF.create_vector_dictionary(DATA_RESSOURCES_PATH + FAST_TEXT_PREFIXOID_VECTORS)
    f9_pref_polarity_dict = PREF.create_polarity_dict(DATA_RESSOURCES_PATH + SENTIMERGE_POLARITY)

    germanet_inventory = {}

    for i in pref_inventory_list:
        f0 = PREF.extract_frequency(i[-3], y_pref_dict, True)  # y_pref_dict or n_pref_dict
        f1 = PREF.transform_class_name_to_binary(i[-1])
        f2 = PREF.extract_frequency(i[0], f2_pref_formations)
        f3 = PREF.extract_frequency(PREF.split_word_at_pipe(i[1])[0], f3_pref_lemmas)
        f4 = PREF.extract_frequency(PREF.split_word_at_pipe(i[1])[1], f4_pref_heads)
        f5 = PREF.calculate_cosine_similarity(i[0], PREF.split_word_at_pipe(i[1])[0], f5_pref_vector_dict)  # reverse for SUFFIXOIDS
        f6 = ''
        i_similar_gn = ''
        i_similar_gn_word = ''
        i_similar_gn_syn = ''
        if i in GN_PREF_FORMATIONS or i in GN_SUFF_FORMATIONS:
            f6 = PREF.search_germanet(i[0], GN_SUPERSENSES, germanet_xml)
        else:
            i_similar_gn = PREF.return_similar_cosine_word(i[0], i[1], f5_pref_vector_dict)
            i_similar_gn_word = PREF.search_germanet(i_similar_gn, GN_SUPERSENSES, germanet_xml, return_single_word=True)
            i_similar_gn_syn = PREF.search_germanet(i_similar_gn_word, GN_SUPERSENSES, germanet_xml)
        if sum(f6) < 1:
            f6 = i_similar_gn_syn
        f7 = PREF.search_germanet(PREF.split_word_at_pipe(i[1])[0], GN_SUPERSENSES, germanet_xml)
        if sum(f7) < 1:
            f7 = i_similar_gn_syn
        f8 = PREF.search_germanet(PREF.split_word_at_pipe(i[1])[1], GN_SUPERSENSES, germanet_xml)
        if sum(f8) < 1:
            f8 = i_similar_gn_syn
        f9 = PREF.extract_polarity(i[0], f9_pref_polarity_dict)
        i_similar_pol = PREF.return_similar_cosine_word(i[0], i[1], f5_pref_vector_dict, False, polarity_dict=f9_pref_polarity_dict)
        i_similar_pol_value = PREF.extract_polarity(i_similar_pol, f9_pref_polarity_dict)
        if f9 == 0:
            f9 = i_similar_pol_value
        f10 = PREF.extract_polarity(PREF.split_word_at_pipe(i[1])[0], f9_pref_polarity_dict)
        if f10 == 0:
            f10 = i_similar_pol_value
        f11 = PREF.extract_polarity(PREF.split_word_at_pipe(i[1])[1], f9_pref_polarity_dict)
        if f11 == 0:
            f11 = i_similar_pol_value
        f0_pref_list.append(f0)
        f1_pref_list.append(f1)
        f2_pref_list.append(f2)
        f3_pref_list.append(f3)
        f4_pref_list.append(f4)
        f5_pref_list.append(f5)
        f6_pref_list.append(f6)
        f7_pref_list.append(f7)
        f8_pref_list.append(f8)
        f9_pref_list.append(f9)
        f10_pref_list.append(f10)
        f11_pref_list.append(f11)
    # print(f0_pref_list)
    # print(len(f0_pref_list))
    # print(f1_pref_list)
    # print(len(f1_pref_list))
    # print(f2_pref_list)
    # print(len(f2_pref_list))
    # print(f3_pref_list)
    # print(len(f3_pref_list))
    # print(f4_pref_list)
    # print(len(f4_pref_list))
    # print(f5_pref_list)
    # print(len(f5_pref_list))
    # print(f6_pref_list)
    # print(len(f6_pref_list))
    # print(f7_pref_list)
    # print(len(f7_pref_list))
    # print(f8_pref_list)
    # print(len(f8_pref_list))
    # print(f9_pref_list)
    # print(len(f9_pref_list))
    # print(f10_pref_list)
    # print(len(f10_pref_list))
    # print(f11_pref_list)
    # print(len(f11_pref_list))

    """Write files"""
    # PREF.write_list_to_file(f0_pref_list, DATA_FEATURES_PATH + 'f0_pref.txt')
    # PREF.write_list_to_file(f1_pref_list, DATA_FEATURES_PATH + 'f1_pref.txt')
    # PREF.write_list_to_file(f2_pref_list, DATA_FEATURES_PATH + 'f2_pref.txt')
    # PREF.write_list_to_file(f3_pref_list, DATA_FEATURES_PATH + 'f3_pref.txt')
    # PREF.write_list_to_file(f4_pref_list, DATA_FEATURES_PATH + 'f4_pref.txt')
    # PREF.write_list_to_file(f5_pref_list, DATA_FEATURES_PATH + 'f5_pref.txt')
    # PREF.write_list_to_file(f6_pref_list, DATA_FEATURES_PATH + 'f6_pref.txt')
    # PREF.write_list_to_file(f7_pref_list, DATA_FEATURES_PATH + 'f7_pref.txt')
    # PREF.write_list_to_file(f8_pref_list, DATA_FEATURES_PATH + 'f8_pref.txt')
    # PREF.write_list_to_file(f9_pref_list, DATA_FEATURES_PATH + 'f9_pref.txt')
    # PREF.write_list_to_file(f10_pref_list, DATA_FEATURES_PATH + 'f10_pref.txt')
    # PREF.write_list_to_file(f11_pref_list, DATA_FEATURES_PATH + 'f11_pref.txt')

    """
        SUFFIXOIDS
    """

    def test_case(word, splitword):
        print(Style.BOLD + word + Style.END)
        t0 = PREF.extract_frequency(PREF.split_word_at_pipe(splitword)[0], y_pref_dict, True)  # change to 1 for suffix
        t1 = PREF.transform_class_name_to_binary('Y')
        t2 = PREF.extract_frequency(word, f2_pref_formations)
        t3 = PREF.extract_frequency(PREF.split_word_at_pipe(splitword)[0], f3_pref_lemmas)
        t4 = PREF.extract_frequency(PREF.split_word_at_pipe(splitword)[1], f4_pref_heads)
        t5 = PREF.calculate_cosine_similarity(word, PREF.split_word_at_pipe(splitword)[0], f5_pref_vector_dict)  # reverse for SUFFIXOIDS
        t6 = PREF.search_germanet(word, GN_SUPERSENSES, germanet_xml)
        t_similar_gn = PREF.return_similar_cosine_word(word, splitword, f5_pref_vector_dict)
        t_similar_gn_word = PREF.search_germanet(t_similar_gn, GN_SUPERSENSES, germanet_xml, return_single_word=True)
        t_similar_gn_syn = PREF.search_germanet(t_similar_gn_word, GN_SUPERSENSES, germanet_xml)
        t7 = PREF.search_germanet(PREF.split_word_at_pipe(splitword)[0], GN_SUPERSENSES, germanet_xml)
        t8 = PREF.search_germanet(PREF.split_word_at_pipe(splitword)[1], GN_SUPERSENSES, germanet_xml)
        t9 = PREF.extract_polarity(word, f9_pref_polarity_dict)
        t_similar_pol = PREF.return_similar_cosine_word(word, splitword, f5_pref_vector_dict, False, polarity_dict=f9_pref_polarity_dict)
        t_similar_pol_value = PREF.extract_polarity(t_similar_pol, f9_pref_polarity_dict)
        t10 = PREF.extract_polarity(PREF.split_word_at_pipe(splitword)[0], f9_pref_polarity_dict)
        t11 = PREF.extract_polarity(PREF.split_word_at_pipe(splitword)[1], f9_pref_polarity_dict)
        print('===================================')
        print('----------TEST CASE----------------')
        print('-----------------------------------')
        print('Candidate vector: ', t0, ' in ', y_pref_dict.keys())
        print('===================================')
        print('Transform class to binary Y: ', t1)
        print('==================================')
        print('Frequency of complex word: ', word, ' = ', t2)
        print('==================================')
        print('Frequency of first part: ', PREF.split_word_at_pipe(splitword)[0], ' = ', t3)
        print('==================================')
        print('Frequency of second part: ', PREF.split_word_at_pipe(splitword)[1], ' = ', t4)
        print('==================================')
        print('Cosine similarity between: ', word, ' and ', PREF.split_word_at_pipe(splitword)[0], ' = ', t5)
        print('==================================')
        print('Searching GermaNet Synsets for: ', word, ' in: ', GN_SUPERSENSES.keys())
        if sum(t6) < 1:
            print('Synset not found')
            print('Similar word: ', t_similar_gn)
            print('Similar synset vector: ', t_similar_gn_syn)
        else:
            print('Synset found')
            print('Vector: ', t6)
        print('==================================')
        print('Searching GermaNet Synsets for: ', PREF.split_word_at_pipe(splitword)[0], ' in: ', GN_SUPERSENSES.keys())
        if sum(t7) < 1:
            print('Synset not found')
            print('Similar word: ', t_similar_gn)
            print('Similar synset vector: ', t_similar_gn_syn)
        else:
            print('Synset found')
            print('Vector: ', t7)
        print('==================================')
        print('Searching GermaNet Synsets for: ', PREF.split_word_at_pipe(splitword)[1], ' in: ',
              GN_SUPERSENSES.keys())
        if sum(t8) < 1:
            print('Synset not found')
            print('Similar word: ', t_similar_gn)
            print('Similar synset vector: ', t_similar_gn_syn)
        else:
            print('Synset found')
            print('Vector: ', t8)
        print('==================================')
        print('Polarity for: ', word)
        if t9 == 0:
            print('Polarity not found')
            print('Similar word from polarity dictionary: ', t_similar_pol)
            print('Similar polarity: ', t_similar_pol_value)
        else:
            print('Polarity: ', t9)
        print('==================================')
        print('Polarity for: ', PREF.split_word_at_pipe(splitword)[0])
        if t10 == 0:
            print('Polarity not found')
            print('Similar word from polarity dictionary: ', t_similar_pol)
            print('Similar polarity: ', t_similar_pol_value)
        else:
            print('Polarity: ', t10)
        print('==================================')
        print('Polarity for: ', PREF.split_word_at_pipe(splitword)[1])
        if t11 == 0:
            print('Polarity not found')
            print('Similar word from polarity dictionary: ', t_similar_pol)
            print('Similar polarity: ', t_similar_pol_value)
        else:
            print('Polarity: ', t11)
        print('==================================')

    # test_case('Bombenattacke', 'Bombe|Attacke')
    # PREF.write_list_to_file(pref_inventory_list, 'out.txt', -1, True)/
    # print(GN_SUPERSENSES)
