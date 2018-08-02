#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
FeatureExtractor

Short description:
This module extracts features from various sources

License: MIT License
Version: 1.0

"""

import ast
import bz2
import io
import json
# import duden
import sys
import math
import matplotlib.pyplot as plt
import numpy as np

from lxml import etree
from sklearn.metrics.pairwise import cosine_similarity

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

""" Frequencies """
FREQUENCY_PREFIXOID_FORMATIONS = 'lemma_frequencies_prefixoid_formations.csv'
FREQUENCY_SUFFIXOID_FORMATIONS = 'lemma_frequencies_suffixoid_formations.csv'
FREQUENCY_PREFIXOID_LEMMAS = 'prefixoid_lemmas_freqs.csv'
FREQUENCY_SUFFIXOID_LEMMAS = 'suffixoid_lemmas_freqs.csv'
FREQUENCY_PREFIXOID_HEADS = 'lemma_frequencies_unique_heads_of_prefixoid_formations.csv'
FREQUENCY_SUFFIXOID_MODIFIERS = 'modifiers_of_suffixoids_lemmas_freqs.csv'

""" fastText vectors """
FAST_TEXT_PREFIXOID_VECTORS = 'fastText/prefixoid-fastText-vectors.txt'
FAST_TEXT_SUFFIXOID_VECTORS = 'fastText/suffixoid-fastText-vectors.txt'

""" Polarity """
SENTIMERGE_POLARITY = 'SentiMerge/sentimerge.txt'

""" Psycholinguistic features; Affective norms """
AFFECTIVE_NORMS = 'AffectiveNorms/ratings_lrec16_koeper_ssiw.txt'

""" Emolex """
EMOLEX = 'EmoLex/NRC-Emotion-Lexicon-v0.92-DE-sorted.csv'

""" PMI Lexicon """
PMI_SCORES = '../../PMI/sdewac_npmi.csv.bz2'
PMI_OUTPUT = 'PMI/'

################
# GermaNet with lxml
################
TREE = etree.parse(DATA_RESSOURCES_PATH+'GermaNet/GN_full.xml')
GN_ROOT = TREE.getroot()
GN_WORDS = GN_ROOT.findall('.//orthForm')

""" GermaNet Supersenses """
GN_SUPERSENSES = {'Allgemein': 0, 'Bewegung': 0, 'Gefuehl': 0, 'Geist': 0, 'Gesellschaft': 0, 'Koerper': 0, 'Menge': 0, 'natPhaenomen': 0, 'Ort': 0, 'Pertonym': 0, 'Perzeption': 0, 'privativ': 0, 'Relation': 0, 'Substanz': 0, 'Verhalten': 0, 'Zeit': 0, 'Artefakt': 0, 'Attribut': 0, 'Besitz': 0, 'Form': 0, 'Geschehen': 0, 'Gruppe': 0, 'Kognition': 0, 'Kommunikation': 0, 'Mensch': 0, 'Motiv': 0, 'Nahrung': 0, 'natGegenstand': 0, 'Pflanze': 0, 'Tier': 0, 'Tops': 0, 'Koerperfunktion': 0, 'Konkurrenz': 0, 'Kontakt': 0, 'Lokation': 0, 'Schoepfung': 0, 'Veraenderung': 0, 'Verbrauch': 0}

""" To prevent unnecessary parsing, use formations already found in GermaNet """
GN_PREF_FORMATIONS = ['Blitzgerät', 'Blitzkarriere', 'Blitzkrieg', 'Blitzkurs', 'Blitzlampe', 'Blitzröhre', 'Blitzschach', 'Blitzschlag', 'Bombenabwurf', 'Bombenalarm', 'Bombendrohung', 'Bombenexplosion', 'Bombenleger', 'Bombennacht', 'Bombenopfer', 'Bombenschacht', 'Bombenschaden', 'Bombensplitter', 'Bombenteppich', 'Bombentest', 'Bombentrichter', 'Glanzente', 'Glanzleistung', 'Glanzlicht', 'Glanzpunkt', 'Glanzrolle', 'Glanzstoff', 'Glanzstück', 'Glanzzeit', 'Jahrhundertfeier', 'Jahrhunderthälfte', 'Jahrhunderthochwasser', 'Jahrhundertsommer', 'Jahrhundertwechsel', 'Qualitätsbewusstsein', 'Qualitätskriterium', 'Qualitätsprüfung', 'Qualitätsstandard', 'Qualitätsverbesserung', 'Qualitätswein', 'Qualitätszuwachs', 'Schweineblut', 'Schweinebraten', 'Schweinefleisch', 'Schweinehaltung', 'Schweinehirte', 'Schweinepest', 'Spitzenfunktionär', 'Spitzenkandidatin', 'Spitzenkoch', 'Spitzenläufer', 'Spitzenleistung', 'Spitzenplatz', 'Spitzenreiter', 'Spitzenspiel', 'Spitzensportler', 'Spitzenverband', 'Spitzenverdiener', 'Spitzenverein', 'Spitzenwert', 'Traummädchen', 'Traumwelt']
GN_SUFF_FORMATIONS = ['Börsenguru', 'Burgunderkönig', 'Bürohengst', 'Dänenkönig', 'Donnergott', 'Dreikönig', 'Feenkönig', 'Feuergott', 'Froschkönig', 'Gegenkönig', 'Gegenpapst', 'Gotenkönig', 'Gottkönig', 'Großkönig', 'Hausschwein', 'Herzkönig', 'Himmelsgott', 'Hochkönig', 'Hunnenkönig', 'Kleinkönig', 'Langobardenkönig', 'Liebesgott', 'Märchenkönig', 'Marienikone', 'Meeresgott', 'Moralapostel', 'Normannenkönig', 'Perserkönig', 'Preußenkönig', 'Priesterkönig', 'Rattenkönig', 'Schlagbolzen', 'Schöpfergott', 'Schützenkönig', 'Schwedenkönig', 'Slawenapostel', 'Sonnenkönig', 'Stammapostel', 'Torschützenkönig', 'Unterkönig', 'Vizekönig', 'Vogelkönig', 'Wachtelkönig', 'Warzenschwein', 'Westgotenkönig', 'Wettergott', 'Wildschwein', 'Winterkönig', 'Zaunkönig', 'Zuchthengst', 'Zwergenkönig']

""" Affixoid dictionary with fastText similarities; sorted """
AFFIXOID_DICTIONARY = 'fastText/affixoid_dict_fasttext_similarites.txt'
PREFIXOID_DICTIONARY = 'fastText/prefixoid_dict_fasttext_similarites.txt'
SUFFIXOID_DICTIONARY = 'fastText/suffixoid_dict_fasttext_similarites.txt'

""" Empty words dictionary for collecting various data """
EMPTY_WORDS_DICTIONARY = 'all_words_dict.txt'


class FeatureExtractor:
    """ FeatureExtractor Class

        Returns: Files with feature vectors

        Example: PREF = FeatureExtractor('Prefixoids', DATA_RESSOURCES_PATH + PREFIXOID_DICTIONARY)

    """

    def __init__(self, string, similar_words_dict):
        print('=' * 40)
        print(Style.BOLD + "Running FeatureExtractor on:" + Style.END, string)
        print('-' * 40)

        try:
            print('Initializing dictionary...')
            self.fasttext_similar_words_dict = self.read_dict_from_file(similar_words_dict)

        except FileNotFoundError:
            print('Please set correct paths for data.')

    def create_affixoid_dictionary(self, affixoid_file, class_name):
        """ This function creates a dictionary with class instances of affixoids

            Args:
                affixoid_file (file): File with affixoid instances
                class_name (str): Class label (Y|N)

            Returns:
                Dictionary with class instances

            Example:
                >>> self.create_affixoid_dictionary(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE, 'Y')

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
                >>> self.read_file_to_list(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE)

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

        print(Style.BOLD + 'File written to:' + Style.END, output_file)

        f.close()

    def write_dict_to_file(self, dictionary, output_file):
        """ Helper function to write a dictionary as string to a file. Import via ast module """

        with io.open(output_file, 'w', encoding='utf8') as data:
            data.write(str(dictionary))

        print(Style.BOLD + 'Dictionary written to:' + Style.END, output_file)

    def read_dict_from_file(self, dictionary_file):
        """ Helper function to read a dictionary from a file. """
        with open(dictionary_file, "rb") as data:
            dictionary = ast.literal_eval(data.read().decode('utf-8'))

        return dictionary

    def read_json_from_file(self, json_file):
        """ Helper function to read a json file. """
        j = open(json_file, 'r', encoding='utf-8')
        json_data = json.load(j)

        return json_data

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
            print(Style.BOLD + 'Class Label not known. Exiting program' + Style.END)
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
            return 0  # return 1?

    def create_frequency_dictionary(self, frequency_file):
        """ This function creates a dictionary with frequency instances of affixoids

            Args:
                frequency_file (file): File with affixoid frequencies

            Returns:
                Dictionary with frequency instances of affixoids

            Example:
                >>> self.create_frequency_dictionary(DATA_PATH + FREQUENCY_PREFIXOID_LEMMAS)

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

    def create_vector_dictionary(self, vector_file, multiword=False):
        """ This function creates a dictionary with vector values from affixoids

            Args:
                vector_file (file): File with vector values from FastText
                multiword (bool): Set to True if the word in vector file has multiple parts

            Returns:
                Dictionary with vector values as list

            Example:
                >>> self.create_vector_dictionary(DATA_RESSOURCES_PATH + FAST_TEXT_PREFIXOID_VECTORS)

        """

        dictionary = {}

        with open(vector_file, 'r', encoding='utf-8') as f:
            for line in f:
                if multiword:
                    word = line.rstrip().split('\t')
                else:
                    word = line.strip().split()
                dict_key = word[0]
                dict_value = list(word[1:])
                dict_value_float = [float(x) for x in dict_value]
                dictionary.update({dict_key: dict_value_float})

        return dictionary

    def create_splitword_dictionary(self, splitword):
        """ Helper function create a dictionary from splitwords """

        dictionary = {}
        dict_key = splitword
        dictionary.update({dict_key: 0})

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

        # adds text for labels, title and axes ticks
        ax.set_ylabel('Counts')
        ax.set_title('Counts per ' + title + ' candidate. Total: ' + str(sum(dict_1.values()) + sum(dict_2.values())) + '')
        ax.set_xticks(ind + width)
        ax.set_xticklabels((dict_1.keys()))

        ax.legend((rects1[0], rects2[0]), ('Y', 'N'))

        def autolabel(rects):
            # attaches text labels
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
            print(Style.BOLD + 'Words not found in fastText vector dictionary' + Style.END)
            return 0

    def search_germanet_supersenses(self, word, fast_text_vector_dict):
        """ This function searches through GermaNet for supersenses noted in GN_SUPERSENSES
            and returns a vector with binary indicator where the sense has been found.
            If the word isn't found, the function calculates cosine similarity between the word
            and a similar word from fastText vector dictionary. If a similar word is found in GermaNet
            the function returns supersenses from the similar word.

            Args:
                word (string): 'Bilderbuchhochzeit'
                fast_text_vector_dict (dict): Dictionary with fastText vectors

            Returns:
                Vector with length of GN_SUPERSENSES.values()

            Example:
                >>> self.search_germanet_supersenses('Husky', {'Tier': 0, 'Mensch': 0})

        """

        print('Searching for: ', word)

        gn_supersenses_dict_copy = GN_SUPERSENSES.copy()

        if self.is_in_germanet(word):
            for synset in GN_ROOT:
                classes = synset.get('class')
                orthforms = synset.findall('.//orthForm')
                for item in orthforms:
                    if word == item.text or word == item.text.lower() or word == item.text.lower().capitalize():
                        print(Style.BOLD + item.text + Style.END, 'supersense >', Style.BOLD + classes + Style.END)
                        gn_supersenses_dict_copy.update({classes: 1})
            return list(gn_supersenses_dict_copy.values())

        else:
            print('Word not found in GermaNet')
            similar_word_supersense = ''
            similar_words_from_fasttext = self.return_similar_words_from_fasttext(word)
            if similar_words_from_fasttext is not None:
                print('Searching for similar words in global fastText dict...')
                for result in similar_words_from_fasttext:
                    print(result)
                    if self.is_in_germanet(result[0]):
                        print(Style.BOLD + result[0] + Style.END, 'found in GermaNet!')
                        supersense = self.search_germanet_supersenses(result[0], fast_text_vector_dict)
                        similar_word_supersense = supersense
                        # return similar_word_supersense
                        break

            if len(similar_word_supersense) > 0:
                print('Word from fastText found in GermaNet...')
                return similar_word_supersense

            else:
                print('Similar words in fastText dict is None or not found')
                print('Lowering threshold and searching for known words with similar cosine')
                similar_word = self.return_similar_cosine_word(word, fast_text_vector_dict)
                supersense = self.search_germanet_supersenses(similar_word, fast_text_vector_dict)
                similar_word_supersense = supersense

            return similar_word_supersense

    def is_in_germanet(self, word):
        """ This function parses GermaNet for a word and returns a boolean if the word is found """

        for item in GN_WORDS:
            if word == item.text or word == item.text.lower() or word == item.text.lower().capitalize():
                return True

        return False

    def is_in_germanet_fast(self, word):
        """ A slightly faster version that parses GermaNet for a word and returns a boolean if the word is found """

        if GN_ROOT.xpath('.//orthForm[text()="'+word+'"]') is not None:
            return True
        else:
            return False

    def return_similar_words_from_fasttext(self, word):
        """ This function returns a list of similar words from fastText """
        if word in self.fasttext_similar_words_dict.keys():
            return self.fasttext_similar_words_dict.get(word)
        else:
            return None

    def return_single_word_from_fasttext(self, word, dictionary):
        """ This function returns a single word from fastText with a given dictionary """

        similar_words_from_fasttext = self.return_similar_words_from_fasttext(word)

        if similar_words_from_fasttext is not None:
            for result in similar_words_from_fasttext:
                if word in dictionary.keys():
                    print(Style.BOLD + result[0] + Style.END, 'found in dictionary!')
                    return result[0]

        return 0

    def return_similar_cosine_word(self, word, fast_text_vector_dict, from_germanet=True, polarity_dict=None, threshold=0.4):
        """ This function calculates cosine similarity between the input word, and all other words
            from fast_text_vector_dict. It returns the most similar word based on cosine similarity.

            Args:
                word (string): 'Bilderbuchhochzeit'
                fast_text_vector_dict (dict): Vector data from a FastText model
                from_germanet (bool): If the given words should also be found in GermaNet.
                polarity_dict dict(): Dictionary with polarities
                threshold (float): Indicator which cosine values to consider.

                # NOTE: higher threshold values may not return a GN word
                # NOTE: set from_germanet to false when using polarity_dict

            Returns:
                Most similar word

            Example:
                >>> self.return_similar_cosine_word('Bilderbuchhochzeit', fast_text_vector_dict, from_germanet=True, threshold=0.4, polarity_dict={})

        """

        print('Searching for similar word to: ', word)

        cosine_similarity_dict = {}

        for key in fast_text_vector_dict.keys():
            if word != key:
                cs_1 = self.calculate_cosine_similarity(word, key, fast_text_vector_dict)
                if cs_1 > threshold:
                    cosine_similarity_dict.update({str(key): cs_1})

        sorted_cosine_similarity_dict = sorted(cosine_similarity_dict.items(), key=lambda kv: kv[1], reverse=True)

        print('Searching in fastText vector dict')

        if from_germanet:
            for x in sorted_cosine_similarity_dict:
                if x[0] in fast_text_vector_dict.keys():
                    if self.is_in_germanet(x[0]):
                        print(Style.BOLD + x[0] + Style.END, 'Found in GermaNet!')
                        print(Style.BOLD + 'Cosine similarity:' + Style.END, x[1])
                        return x[0]

        else:
            print('Searching in PolarityDict')
            for x in sorted_cosine_similarity_dict:
                if x[0] in polarity_dict.keys():
                    print(Style.BOLD + x[0] + Style.END, 'Found in PolarityDict!')
                    print(Style.BOLD + 'Cosine similarity:' + Style.END, x[1])
                    return x[0]

    def create_polarity_dict(self, polarity_file):
        """ Helper function to create a polarity dictionary, where key = word and value = [vector of values] """

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

    def extract_dictionary_values(self, word, polarity_dict):
        """ Helper function to extract polarity for a word from dictionary """

        if word in polarity_dict.keys():
            value = polarity_dict[word]
            try:
                v = ast.literal_eval(value)
                return v
            except ValueError:
                return value

        else:
            return 0

    def extract_pmi_values(self, splitwords_dictionary, output_file):
        """ Helper function to extract PMI values for a word from a bz2 file and write values to a file """

        print('Extracting PMI scores...')

        with bz2.BZ2File(PMI_SCORES, 'r') as pmi_file:
            for line in pmi_file:
                words = line.split()
                decoded = []
                for w in words:
                    word = w.decode('UTF-8')
                    decoded.append(word)
                try:
                    splitword = str(decoded[0]) + '|' + str(decoded[1])
                    if splitword in splitwords_dictionary.keys():
                        print('Found!', decoded)
                        splitwords_dictionary.update({splitword: [decoded[2], decoded[3]]})
                except:
                    pass

        self.write_dict_to_file(splitwords_dictionary, output_file)

        return splitwords_dictionary


class Style:
    """ Helper class for nicer coloring """
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    UNDERLINE = '\033[4m'
    BOLD = '\033[1m'
    END = '\033[0m'


def sigmoid(x):
    """ Helper function to calculate sigmoid value for x """
    return 1 / (1 + math.exp(-x))


def min_max_scaling(x, min_x, max_x):
    """ Helper function to calculate min/max scaling for x """
    return (x - min_x) / (max_x - min_x)


if __name__ == "__main__":
    """
        PREFIXOIDS
    """
    PREF = FeatureExtractor('Prefixoids', DATA_RESSOURCES_PATH + PREFIXOID_DICTIONARY)
    pref_inventory_list = PREF.read_file_to_list(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE)
    y_pref_dict = PREF.create_affixoid_dictionary(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE, 'Y')
    n_pref_dict = PREF.create_affixoid_dictionary(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE, 'N')
    print('Y:\t', y_pref_dict)
    print('N:\t', n_pref_dict)
    print('Total:\t', sum(y_pref_dict.values()) + sum(n_pref_dict.values()))
    print(len(pref_inventory_list))
    # PREF.plot_statistics(y_pref_dict, n_pref_dict, 'Prefixoids')

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
    f12_pref_list = []  # Affective Norms for complex word
    f13_pref_list = []  # Affective Norms for first part
    f14_pref_list = []  # Affective Norms for second part
    f15_pref_list = []  # Emotion for complex word
    f16_pref_list = []  # Emotion for first part
    f17_pref_list = []  # Emotion for second part

    f2_pref_formations = PREF.create_frequency_dictionary(DATA_PATH + FREQUENCY_PREFIXOID_FORMATIONS)
    f3_pref_lemmas = PREF.create_frequency_dictionary(DATA_PATH + FREQUENCY_PREFIXOID_LEMMAS)
    f4_pref_heads = PREF.create_frequency_dictionary(DATA_PATH + FREQUENCY_PREFIXOID_HEADS)
    f5_pref_vector_dict = PREF.create_vector_dictionary(DATA_RESSOURCES_PATH + FAST_TEXT_PREFIXOID_VECTORS)
    f9_pref_polarity_dict = PREF.create_polarity_dict(DATA_RESSOURCES_PATH + SENTIMERGE_POLARITY)
    f12_pref_affective_norms_dict = PREF.create_vector_dictionary(DATA_RESSOURCES_PATH + AFFECTIVE_NORMS)
    f15_pref_emolex_dict = PREF.create_vector_dictionary(DATA_RESSOURCES_PATH + EMOLEX, multiword=True)
    f18_pref_splitwords_dict = {}

    maximum_f2_pref_formations = max(f2_pref_formations, key=f2_pref_formations.get)
    maximum_f3_pref_lemmas = max(f3_pref_lemmas, key=f3_pref_lemmas.get)
    maximum_f4_pref_heads = max(f4_pref_heads, key=f4_pref_heads.get)

    print('Max frequencies')
    print('Formations:', f2_pref_formations[maximum_f2_pref_formations],
          'Lemmas:', f3_pref_lemmas[maximum_f3_pref_lemmas],
          'Heads:', f4_pref_heads[maximum_f4_pref_heads])
    # Formations: 97 Lemmas: 94232 Heads: 9988

    counter = 0
    for i in pref_inventory_list:
        counter += 1
        print('Line:', str(counter) + ' ===============================', i[0], i[-1])

        f0 = PREF.extract_frequency(i[-3], y_pref_dict, True)  # y_pref_dict or n_pref_dict
        f1 = PREF.transform_class_name_to_binary(i[-1])
        f2 = PREF.extract_frequency(i[0], f2_pref_formations)
        f3 = PREF.extract_frequency(PREF.split_word_at_pipe(i[1])[0], f3_pref_lemmas)
        f4 = PREF.extract_frequency(PREF.split_word_at_pipe(i[1])[1], f4_pref_heads)
        f5 = PREF.calculate_cosine_similarity(i[0], PREF.split_word_at_pipe(i[1])[0], f5_pref_vector_dict)  # split_word_at_pipe(i[1])[1] for SUFFIXOIDS
        f6 = PREF.search_germanet_supersenses(i[0], f5_pref_vector_dict)
        f7 = PREF.search_germanet_supersenses(PREF.split_word_at_pipe(i[1])[0], f5_pref_vector_dict)
        f8 = PREF.search_germanet_supersenses(PREF.split_word_at_pipe(i[1])[1], f5_pref_vector_dict)

        f9 = PREF.extract_dictionary_values(i[0], f9_pref_polarity_dict)
        if f9 == 0:
            f9_similar_pol = PREF.return_single_word_from_fasttext(i[0], f9_pref_polarity_dict)
            if f9_similar_pol == 0:
                f9_similar_pol = PREF.return_similar_cosine_word(i[0], f5_pref_vector_dict, False, polarity_dict=f9_pref_polarity_dict)
                f9_similar_pol_value = PREF.extract_dictionary_values(f9_similar_pol, f9_pref_polarity_dict)
            else:
                f9_similar_pol_value = PREF.extract_dictionary_values(f9_similar_pol, f9_pref_polarity_dict)
            f9 = f9_similar_pol_value

        f10 = PREF.extract_dictionary_values(PREF.split_word_at_pipe(i[1])[0], f9_pref_polarity_dict)
        if f10 == 0:
            f10_similar_pol = PREF.return_single_word_from_fasttext(PREF.split_word_at_pipe(i[1])[0], f9_pref_polarity_dict)
            if f10_similar_pol == 0:
                f10_similar_pol = PREF.return_similar_cosine_word(PREF.split_word_at_pipe(i[1])[0], f5_pref_vector_dict, False, polarity_dict=f9_pref_polarity_dict)
                f10_similar_pol_value = PREF.extract_dictionary_values(f10_similar_pol, f9_pref_polarity_dict)
            else:
                f10_similar_pol_value = PREF.extract_dictionary_values(f10_similar_pol, f9_pref_polarity_dict)
            f10 = f10_similar_pol_value

        f11 = PREF.extract_dictionary_values(PREF.split_word_at_pipe(i[1])[1], f9_pref_polarity_dict)
        if f11 == 0:
            f11_similar_pol = PREF.return_single_word_from_fasttext(PREF.split_word_at_pipe(i[1])[1], f9_pref_polarity_dict)
            if f11_similar_pol == 0:
                f11_similar_pol = PREF.return_similar_cosine_word(PREF.split_word_at_pipe(i[1])[1], f5_pref_vector_dict, False, polarity_dict=f9_pref_polarity_dict)
                f11_similar_pol_value = PREF.extract_dictionary_values(f11_similar_pol, f9_pref_polarity_dict)
            else:
                f11_similar_pol_value = PREF.extract_dictionary_values(f11_similar_pol, f9_pref_polarity_dict)
            f11 = f11_similar_pol_value


        f12 = PREF.extract_dictionary_values(i[0], f12_pref_affective_norms_dict)
        if f12 == 0:
            f12_similar_pol = PREF.return_single_word_from_fasttext(i[0], f12_pref_affective_norms_dict)
            if f12_similar_pol == 0:
                f12_similar_pol = PREF.return_similar_cosine_word(i[0], f5_pref_vector_dict, False, polarity_dict=f12_pref_affective_norms_dict)
                f12_similar_pol_value = PREF.extract_dictionary_values(f12_similar_pol, f12_pref_affective_norms_dict)
            else:
                f12_similar_pol_value = PREF.extract_dictionary_values(f12_similar_pol, f12_pref_affective_norms_dict)
            f12 = f12_similar_pol_value

        f13 = PREF.extract_dictionary_values(PREF.split_word_at_pipe(i[1])[0], f12_pref_affective_norms_dict)
        if f13 == 0:
            f13_similar_pol = PREF.return_single_word_from_fasttext(PREF.split_word_at_pipe(i[1])[0], f12_pref_affective_norms_dict)
            if f13_similar_pol == 0:
                f13_similar_pol = PREF.return_similar_cosine_word(PREF.split_word_at_pipe(i[1])[0], f5_pref_vector_dict, False, polarity_dict=f12_pref_affective_norms_dict)
                f13_similar_pol_value = PREF.extract_dictionary_values(f13_similar_pol, f12_pref_affective_norms_dict)
            else:
                f13_similar_pol_value = PREF.extract_dictionary_values(f13_similar_pol, f12_pref_affective_norms_dict)
            f13 = f13_similar_pol_value

        f14 = PREF.extract_dictionary_values(PREF.split_word_at_pipe(i[1])[1], f12_pref_affective_norms_dict)
        if f14 == 0:
            f14_similar_pol = PREF.return_single_word_from_fasttext(PREF.split_word_at_pipe(i[1])[1], f12_pref_affective_norms_dict)
            if f14_similar_pol == 0:
                f14_similar_pol = PREF.return_similar_cosine_word(PREF.split_word_at_pipe(i[1])[1], f5_pref_vector_dict, False, polarity_dict=f12_pref_affective_norms_dict)
                f14_similar_pol_value = PREF.extract_dictionary_values(f14_similar_pol, f12_pref_affective_norms_dict)
            else:
                f14_similar_pol_value = PREF.extract_dictionary_values(f14_similar_pol, f12_pref_affective_norms_dict)
            f14 = f14_similar_pol_value

        f15 = PREF.extract_dictionary_values(i[0], f15_pref_emolex_dict)
        if f15 == 0:
            f15_similar_pol = PREF.return_single_word_from_fasttext(i[0], f15_pref_emolex_dict)
            if f15_similar_pol == 0:
                f15_similar_pol = PREF.return_similar_cosine_word(i[0], f5_pref_vector_dict, False, polarity_dict=f15_pref_emolex_dict)
                f15_similar_pol_value = PREF.extract_dictionary_values(f15_similar_pol, f15_pref_emolex_dict)
            else:
                f15_similar_pol_value = PREF.extract_dictionary_values(f15_similar_pol, f15_pref_emolex_dict)
            f15 = f15_similar_pol_value

        f16 = PREF.extract_dictionary_values(PREF.split_word_at_pipe(i[1])[0], f15_pref_emolex_dict)
        if f16 == 0:
            f16_similar_pol = PREF.return_single_word_from_fasttext(PREF.split_word_at_pipe(i[1])[0], f15_pref_emolex_dict)
            if f16_similar_pol == 0:
                f16_similar_pol = PREF.return_similar_cosine_word(PREF.split_word_at_pipe(i[1])[0], f5_pref_vector_dict, False, polarity_dict=f15_pref_emolex_dict)
                f16_similar_pol_value = PREF.extract_dictionary_values(f16_similar_pol, f15_pref_emolex_dict)
            else:
                f16_similar_pol_value = PREF.extract_dictionary_values(f16_similar_pol, f15_pref_emolex_dict)
            f16 = f16_similar_pol_value

        f17 = PREF.extract_dictionary_values(PREF.split_word_at_pipe(i[1])[1], f15_pref_emolex_dict)
        if f17 == 0:
            f17_similar_pol = PREF.return_single_word_from_fasttext(PREF.split_word_at_pipe(i[1])[1], f15_pref_emolex_dict)
            if f17_similar_pol == 0:
                f17_similar_pol = PREF.return_similar_cosine_word(PREF.split_word_at_pipe(i[1])[1], f5_pref_vector_dict, False, polarity_dict=f15_pref_emolex_dict)
                f17_similar_pol_value = PREF.extract_dictionary_values(f17_similar_pol, f15_pref_emolex_dict)
            else:
                f17_similar_pol_value = PREF.extract_dictionary_values(f17_similar_pol, f15_pref_emolex_dict)
            f17 = f17_similar_pol_value

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
        f12_pref_list.append(f12)
        f13_pref_list.append(f13)
        f14_pref_list.append(f14)
        f15_pref_list.append(f15)
        f16_pref_list.append(f16)
        f17_pref_list.append(f17)

        f18_pref_splitwords_dict.update({i[1]: [10 ** -3, 10 ** -3]})

    # print(f0_pref_list)
    print(len(f0_pref_list))
    # print(f1_pref_list)
    print(len(f1_pref_list))
    # print(f2_pref_list)
    print(len(f2_pref_list))
    # print(f3_pref_list)
    print(len(f3_pref_list))
    # print(f4_pref_list)
    print(len(f4_pref_list))
    # print(f5_pref_list)
    print(len(f5_pref_list))
    # print(f6_pref_list)
    print(len(f6_pref_list))
    # print(f7_pref_list)
    print(len(f7_pref_list))
    # print(f8_pref_list)
    print(len(f8_pref_list))
    # print(f9_pref_list)
    print(len(f9_pref_list))
    # print(f10_pref_list)
    print(len(f10_pref_list))
    # print(f11_pref_list)
    print(len(f11_pref_list))
    # print(f12_pref_list)
    print(len(f12_pref_list))
    # print(f13_pref_list)
    print(len(f13_pref_list))
    # print(f14_pref_list)
    print(len(f14_pref_list))
    # print(f15_pref_list)
    print(len(f15_pref_list))
    # print(f16_pref_list)
    print(len(f16_pref_list))
    # print(f17_pref_list)
    print(len(f17_pref_list))

    """ Write files """
    PREF.write_list_to_file(f0_pref_list, DATA_FEATURES_PATH + 'f0_pref.txt')  # DONE
    PREF.write_list_to_file(f1_pref_list, DATA_FEATURES_PATH + 'f1_pref.txt')  # DONE
    PREF.write_list_to_file(f2_pref_list, DATA_FEATURES_PATH + 'f2_pref.txt')  # DONE
    PREF.write_list_to_file(f3_pref_list, DATA_FEATURES_PATH + 'f3_pref.txt')  # DONE
    PREF.write_list_to_file(f4_pref_list, DATA_FEATURES_PATH + 'f4_pref.txt')  # DONE
    PREF.write_list_to_file(f5_pref_list, DATA_FEATURES_PATH + 'f5_pref.txt')  # DONE
    PREF.write_list_to_file(f6_pref_list, DATA_FEATURES_PATH + 'f6_pref.txt')  # DONE
    PREF.write_list_to_file(f7_pref_list, DATA_FEATURES_PATH + 'f7_pref.txt')  # DONE
    PREF.write_list_to_file(f8_pref_list, DATA_FEATURES_PATH + 'f8_pref.txt')  # DONE
    PREF.write_list_to_file(f9_pref_list, DATA_FEATURES_PATH + 'f9_pref.txt')  # DONE
    PREF.write_list_to_file(f10_pref_list, DATA_FEATURES_PATH + 'f10_pref.txt')  # DONE
    PREF.write_list_to_file(f11_pref_list, DATA_FEATURES_PATH + 'f11_pref.txt')  # DONE
    PREF.write_list_to_file(f12_pref_list, DATA_FEATURES_PATH + 'f12_pref.txt')  # DONE
    PREF.write_list_to_file(f13_pref_list, DATA_FEATURES_PATH + 'f13_pref.txt')  # DONE
    PREF.write_list_to_file(f14_pref_list, DATA_FEATURES_PATH + 'f14_pref.txt')  # DONE
    PREF.write_list_to_file(f15_pref_list, DATA_FEATURES_PATH + 'f15_pref.txt')  # DONE
    PREF.write_list_to_file(f16_pref_list, DATA_FEATURES_PATH + 'f16_pref.txt')  # DONE
    PREF.write_list_to_file(f17_pref_list, DATA_FEATURES_PATH + 'f17_pref.txt')  # DONE

    """ PMI Scores """
    f18_pref_pmi_dict = PREF.extract_pmi_values(f18_pref_splitwords_dict, DATA_RESSOURCES_PATH + PMI_OUTPUT + 'pref_PMI_scores.txt')
    f18_pref_list = []  # PMI Scores for first and second part of word

    """ Second loop over inventory (after collecting splitwords) """
    for i in pref_inventory_list:
        f18 = PREF.extract_dictionary_values(i[1], f18_pref_pmi_dict)
        f18_pref_list.append(f18)

    # print(f18_pref_list)
    print(len(f18_pref_list))

    PREF.write_list_to_file(f18_pref_list, DATA_FEATURES_PATH + 'f18_pref.txt')

    """
        SUFFIXOIDS
    """
    SUFF = FeatureExtractor('Suffixoids', DATA_RESSOURCES_PATH + SUFFIXOID_DICTIONARY)
    suff_inventory_list = SUFF.read_file_to_list(DATA_FINAL_PATH + FINAL_SUFFIXOID_FILE)
    y_suff_dict = SUFF.create_affixoid_dictionary(DATA_FINAL_PATH + FINAL_SUFFIXOID_FILE, 'Y')
    n_suff_dict = SUFF.create_affixoid_dictionary(DATA_FINAL_PATH + FINAL_SUFFIXOID_FILE, 'N')
    print('Y:\t', y_suff_dict)
    print('N:\t', n_suff_dict)
    print('Total:\t', sum(y_suff_dict.values()) + sum(n_suff_dict.values()))
    print(len(suff_inventory_list))
    # SUFF.plot_statistics(y_suff_dict, n_suff_dict, 'Suffixoids')

    f0_suff_list = []  # prefix coded into dictionary
    f1_suff_list = []  # binary indicator, if affixoid
    f2_suff_list = []  # frequency of complex word
    f3_suff_list = []  # frequency of first part
    f4_suff_list = []  # frequency of second part
    f5_suff_list = []  # cosine similarity between complex word and head
    f6_suff_list = []  # vector of GermaNet supersenses for complex word
    f7_suff_list = []  # vector of GermaNet supersenses for first part
    f8_suff_list = []  # vector of GermaNet supersenses for second part
    f9_suff_list = []  # SentiMerge Polarity for complex word
    f10_suff_list = []  # SentiMerge Polarity for first part
    f11_suff_list = []  # SentiMerge Polarity for second part
    f12_suff_list = []  # Affective Norms for complex word
    f13_suff_list = []  # Affective Norms for first part
    f14_suff_list = []  # Affective Norms for second part
    f15_suff_list = []  # Emotion for complex word
    f16_suff_list = []  # Emotion for first part
    f17_suff_list = []  # Emotion for second part

    f2_suff_formations = SUFF.create_frequency_dictionary(DATA_PATH + FREQUENCY_SUFFIXOID_FORMATIONS)
    f3_suff_lemmas = SUFF.create_frequency_dictionary(DATA_PATH + FREQUENCY_SUFFIXOID_MODIFIERS)
    f4_suff_heads = SUFF.create_frequency_dictionary(DATA_PATH + FREQUENCY_SUFFIXOID_LEMMAS)
    f5_suff_vector_dict = SUFF.create_vector_dictionary(DATA_RESSOURCES_PATH + FAST_TEXT_SUFFIXOID_VECTORS)
    f9_suff_polarity_dict = SUFF.create_polarity_dict(DATA_RESSOURCES_PATH + SENTIMERGE_POLARITY)
    f12_suff_affective_norms_dict = SUFF.create_vector_dictionary(DATA_RESSOURCES_PATH + AFFECTIVE_NORMS)
    f15_suff_emolex_dict = SUFF.create_vector_dictionary(DATA_RESSOURCES_PATH + EMOLEX, multiword=True)
    f18_suff_splitwords_dict = {}

    maximum_f2_suff_formations = max(f2_suff_formations, key=f2_suff_formations.get)
    maximum_f3_suff_lemmas = max(f3_suff_lemmas, key=f3_suff_lemmas.get)
    maximum_f4_suff_heads = max(f4_suff_heads, key=f4_suff_heads.get)

    print('Max frequencies')
    print('Formations:', f2_suff_formations[maximum_f2_suff_formations],
          'Lemmas:', f3_suff_lemmas[maximum_f3_suff_lemmas],
          'Heads:', f4_suff_heads[maximum_f4_suff_heads])
    # Formations: 99 Lemmas: 931 Heads: 998048

    counter = 0
    for i in suff_inventory_list:
        counter += 1
        print('Line:', str(counter) + ' ===============================', i[0], i[-1])

        f0 = SUFF.extract_frequency(i[-3], y_suff_dict, True)  # y_suff_dict or n_suff_dict
        f1 = SUFF.transform_class_name_to_binary(i[-1])
        f2 = SUFF.extract_frequency(i[0], f2_suff_formations)
        f3 = SUFF.extract_frequency(SUFF.split_word_at_pipe(i[1])[0], f3_suff_lemmas)
        f4 = SUFF.extract_frequency(SUFF.split_word_at_pipe(i[1])[1], f4_suff_heads)
        f5 = SUFF.calculate_cosine_similarity(i[0], SUFF.split_word_at_pipe(i[1])[1], f5_suff_vector_dict)  # split_word_at_pipe(i[1])[0] for PREFIXOIDS
        f6 = SUFF.search_germanet_supersenses(i[0], f5_suff_vector_dict)
        f7 = SUFF.search_germanet_supersenses(SUFF.split_word_at_pipe(i[1])[0], f5_suff_vector_dict)
        f8 = SUFF.search_germanet_supersenses(SUFF.split_word_at_pipe(i[1])[1], f5_suff_vector_dict)

        f9 = SUFF.extract_dictionary_values(i[0], f9_suff_polarity_dict)
        if f9 == 0:
            f9_similar_pol = SUFF.return_single_word_from_fasttext(i[0], f9_suff_polarity_dict)
            if f9_similar_pol == 0:
                f9_similar_pol = SUFF.return_similar_cosine_word(i[0], f5_suff_vector_dict, False, polarity_dict=f9_suff_polarity_dict)
                f9_similar_pol_value = SUFF.extract_dictionary_values(f9_similar_pol, f9_suff_polarity_dict)
            else:
                f9_similar_pol_value = SUFF.extract_dictionary_values(f9_similar_pol, f9_suff_polarity_dict)
            f9 = f9_similar_pol_value

        f10 = SUFF.extract_dictionary_values(SUFF.split_word_at_pipe(i[1])[0], f9_suff_polarity_dict)
        if f10 == 0:
            f10_similar_pol = SUFF.return_single_word_from_fasttext(SUFF.split_word_at_pipe(i[1])[0], f9_suff_polarity_dict)
            if f10_similar_pol == 0:
                f10_similar_pol = SUFF.return_similar_cosine_word(SUFF.split_word_at_pipe(i[1])[0], f5_suff_vector_dict, False, polarity_dict=f9_suff_polarity_dict)
                f10_similar_pol_value = SUFF.extract_dictionary_values(f10_similar_pol, f9_suff_polarity_dict)
            else:
                f10_similar_pol_value = SUFF.extract_dictionary_values(f10_similar_pol, f9_suff_polarity_dict)
            f10 = f10_similar_pol_value

        f11 = SUFF.extract_dictionary_values(SUFF.split_word_at_pipe(i[1])[1], f9_suff_polarity_dict)
        if f11 == 0:
            f11_similar_pol = SUFF.return_single_word_from_fasttext(SUFF.split_word_at_pipe(i[1])[1], f9_suff_polarity_dict)
            if f11_similar_pol == 0:
                f11_similar_pol = SUFF.return_similar_cosine_word(SUFF.split_word_at_pipe(i[1])[1], f5_suff_vector_dict, False, polarity_dict=f9_suff_polarity_dict)
                f11_similar_pol_value = SUFF.extract_dictionary_values(f11_similar_pol, f9_suff_polarity_dict)
            else:
                f11_similar_pol_value = SUFF.extract_dictionary_values(f11_similar_pol, f9_suff_polarity_dict)
            f11 = f11_similar_pol_value

        f12 = SUFF.extract_dictionary_values(i[0], f12_suff_affective_norms_dict)
        if f12 == 0:
            f12_similar_pol = SUFF.return_single_word_from_fasttext(i[0], f12_suff_affective_norms_dict)
            if f12_similar_pol == 0:
                f12_similar_pol = SUFF.return_similar_cosine_word(i[0], f5_suff_vector_dict, False, polarity_dict=f12_suff_affective_norms_dict)
                f12_similar_pol_value = SUFF.extract_dictionary_values(f12_similar_pol, f12_suff_affective_norms_dict)
            else:
                f12_similar_pol_value = SUFF.extract_dictionary_values(f12_similar_pol, f12_suff_affective_norms_dict)
            f12 = f12_similar_pol_value

        f13 = SUFF.extract_dictionary_values(SUFF.split_word_at_pipe(i[1])[0], f12_suff_affective_norms_dict)
        if f13 == 0:
            f13_similar_pol = SUFF.return_single_word_from_fasttext(SUFF.split_word_at_pipe(i[1])[0], f12_suff_affective_norms_dict)
            if f13_similar_pol == 0:
                f13_similar_pol = SUFF.return_similar_cosine_word(SUFF.split_word_at_pipe(i[1])[0], f5_suff_vector_dict, False, polarity_dict=f12_suff_affective_norms_dict)
                f13_similar_pol_value = SUFF.extract_dictionary_values(f13_similar_pol, f12_suff_affective_norms_dict)
            else:
                f13_similar_pol_value = SUFF.extract_dictionary_values(f13_similar_pol, f12_suff_affective_norms_dict)
            f13 = f13_similar_pol_value

        f14 = SUFF.extract_dictionary_values(SUFF.split_word_at_pipe(i[1])[1], f12_suff_affective_norms_dict)
        if f14 == 0:
            f14_similar_pol = SUFF.return_single_word_from_fasttext(SUFF.split_word_at_pipe(i[1])[1], f12_suff_affective_norms_dict)
            if f14_similar_pol == 0:
                f14_similar_pol = SUFF.return_similar_cosine_word(SUFF.split_word_at_pipe(i[1])[1], f5_suff_vector_dict, False, polarity_dict=f12_suff_affective_norms_dict)
                f14_similar_pol_value = SUFF.extract_dictionary_values(f14_similar_pol, f12_suff_affective_norms_dict)
            else:
                f14_similar_pol_value = SUFF.extract_dictionary_values(f14_similar_pol, f12_suff_affective_norms_dict)
            f14 = f14_similar_pol_value

        f15 = SUFF.extract_dictionary_values(i[0], f15_suff_emolex_dict)
        if f15 == 0:
            f15_similar_pol = SUFF.return_single_word_from_fasttext(i[0], f15_suff_emolex_dict)
            if f15_similar_pol == 0:
                f15_similar_pol = SUFF.return_similar_cosine_word(i[0], f5_suff_vector_dict, False, polarity_dict=f15_suff_emolex_dict)
                f15_similar_pol_value = SUFF.extract_dictionary_values(f15_similar_pol, f15_suff_emolex_dict)
            else:
                f15_similar_pol_value = SUFF.extract_dictionary_values(f15_similar_pol, f15_suff_emolex_dict)
            f15 = f15_similar_pol_value

        f16 = SUFF.extract_dictionary_values(SUFF.split_word_at_pipe(i[1])[0], f15_suff_emolex_dict)
        if f16 == 0:
            f16_similar_pol = SUFF.return_single_word_from_fasttext(SUFF.split_word_at_pipe(i[1])[0], f15_suff_emolex_dict)
            if f16_similar_pol == 0:
                f16_similar_pol = SUFF.return_similar_cosine_word(SUFF.split_word_at_pipe(i[1])[0], f5_suff_vector_dict, False, polarity_dict=f15_suff_emolex_dict)
                f16_similar_pol_value = SUFF.extract_dictionary_values(f16_similar_pol, f15_suff_emolex_dict)
            else:
                f16_similar_pol_value = SUFF.extract_dictionary_values(f16_similar_pol, f15_suff_emolex_dict)
            f16 = f16_similar_pol_value

        f17 = SUFF.extract_dictionary_values(SUFF.split_word_at_pipe(i[1])[1], f15_suff_emolex_dict)
        if f17 == 0:
            f17_similar_pol = SUFF.return_single_word_from_fasttext(SUFF.split_word_at_pipe(i[1])[1], f15_suff_emolex_dict)
            if f17_similar_pol == 0:
                f17_similar_pol = SUFF.return_similar_cosine_word(SUFF.split_word_at_pipe(i[1])[1], f5_suff_vector_dict, False, polarity_dict=f15_suff_emolex_dict)
                f17_similar_pol_value = SUFF.extract_dictionary_values(f17_similar_pol, f15_suff_emolex_dict)
            else:
                f17_similar_pol_value = SUFF.extract_dictionary_values(f17_similar_pol, f15_suff_emolex_dict)
            f17 = f17_similar_pol_value

        f0_suff_list.append(f0)
        f1_suff_list.append(f1)
        f2_suff_list.append(f2)
        f3_suff_list.append(f3)
        f4_suff_list.append(f4)
        f5_suff_list.append(f5)
        f6_suff_list.append(f6)
        f7_suff_list.append(f7)
        f8_suff_list.append(f8)
        f9_suff_list.append(f9)
        f10_suff_list.append(f10)
        f11_suff_list.append(f11)
        f12_suff_list.append(f12)
        f13_suff_list.append(f13)
        f14_suff_list.append(f14)
        f15_suff_list.append(f15)
        f16_suff_list.append(f16)
        f17_suff_list.append(f17)

        f18_suff_splitwords_dict.update({i[1]: [10 ** -3, 10 ** -3]})

    # print(f0_suff_list)
    print(len(f0_suff_list))
    # print(f1_suff_list)
    print(len(f1_suff_list))
    # print(f2_suff_list)
    print(len(f2_suff_list))
    # print(f3_suff_list)
    print(len(f3_suff_list))
    # print(f4_suff_list)
    print(len(f4_suff_list))
    # print(f5_suff_list)
    print(len(f5_suff_list))
    # print(f6_suff_list)
    print(len(f6_suff_list))
    # print(f7_suff_list)
    print(len(f7_suff_list))
    # print(f8_suff_list)
    print(len(f8_suff_list))
    # print(f9_suff_list)
    print(len(f9_suff_list))
    # print(f10_suff_list)
    print(len(f10_suff_list))
    # print(f11_suff_list)
    print(len(f11_suff_list))
    # print(f12_suff_list)
    print(len(f12_suff_list))
    # print(f13_suff_list)
    print(len(f13_suff_list))
    # print(f14_suff_list)
    print(len(f14_suff_list))
    # print(f15_suff_list)
    print(len(f15_suff_list))
    # print(f16_suff_list)
    print(len(f16_suff_list))
    # print(f17_suff_list)
    print(len(f17_suff_list))

    """ Write files """
    SUFF.write_list_to_file(f0_suff_list, DATA_FEATURES_PATH + 'f0_suff.txt')  # DONE
    SUFF.write_list_to_file(f1_suff_list, DATA_FEATURES_PATH + 'f1_suff.txt')  # DONE
    SUFF.write_list_to_file(f2_suff_list, DATA_FEATURES_PATH + 'f2_suff.txt')  # DONE
    SUFF.write_list_to_file(f3_suff_list, DATA_FEATURES_PATH + 'f3_suff.txt')  # DONE
    SUFF.write_list_to_file(f4_suff_list, DATA_FEATURES_PATH + 'f4_suff.txt')  # DONE
    SUFF.write_list_to_file(f5_suff_list, DATA_FEATURES_PATH + 'f5_suff.txt')  # DONE
    SUFF.write_list_to_file(f6_suff_list, DATA_FEATURES_PATH + 'f6_suff.txt')  # DONE
    SUFF.write_list_to_file(f7_suff_list, DATA_FEATURES_PATH + 'f7_suff.txt')  # DONE
    SUFF.write_list_to_file(f8_suff_list, DATA_FEATURES_PATH + 'f8_suff.txt')  # DONE
    SUFF.write_list_to_file(f9_suff_list, DATA_FEATURES_PATH + 'f9_suff.txt')  # DONE
    SUFF.write_list_to_file(f10_suff_list, DATA_FEATURES_PATH + 'f10_suff.txt')  # DONE
    SUFF.write_list_to_file(f11_suff_list, DATA_FEATURES_PATH + 'f11_suff.txt')  # DONE
    SUFF.write_list_to_file(f12_suff_list, DATA_FEATURES_PATH + 'f12_suff.txt')  # DONE
    SUFF.write_list_to_file(f13_suff_list, DATA_FEATURES_PATH + 'f13_suff.txt')  # DONE
    SUFF.write_list_to_file(f14_suff_list, DATA_FEATURES_PATH + 'f14_suff.txt')  # DONE
    SUFF.write_list_to_file(f15_suff_list, DATA_FEATURES_PATH + 'f15_suff.txt')  # DONE
    SUFF.write_list_to_file(f16_suff_list, DATA_FEATURES_PATH + 'f16_suff.txt')  # DONE
    SUFF.write_list_to_file(f17_suff_list, DATA_FEATURES_PATH + 'f17_suff.txt')  # DONE

    """ PMI Scores """
    f18_suff_pmi_dict = SUFF.extract_pmi_values(f18_suff_splitwords_dict, DATA_RESSOURCES_PATH + PMI_OUTPUT + 'suff_PMI_scores.txt')
    f18_suff_list = []  # PMI Scores for first and second part of word

    """ Second loop over inventory (after collecting splitwords) """
    for i in suff_inventory_list:
        f18 = SUFF.extract_dictionary_values(i[1], f18_suff_pmi_dict)
        f18_suff_list.append(f18)

    # print(f18_suff_list)
    print(len(f18_suff_list))

    SUFF.write_list_to_file(f18_suff_list, DATA_FEATURES_PATH + 'f18_suff.txt')
