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

# import duden
import configparser
import math
import matplotlib.pyplot as plt
import numpy as np

from modules import dictionaries as dc
from modules import file_writer as fw
from modules import file_reader as fr
from modules import helper_functions as hf

from lxml import etree
from sklearn.metrics.pairwise import cosine_similarity

########################
# GLOBAL FILE SETTINGS
########################
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('config.ini')

################
# GermaNet with lxml
################
TREE = etree.parse(config.get('PathSettings', 'RessourcesPath')+'GermaNet/GN_full.xml')
GN_ROOT = TREE.getroot()
GN_WORDS = GN_ROOT.findall('.//orthForm')

""" GermaNet Supersenses """
GN_SUPERSENSES = {'Allgemein': 0, 'Bewegung': 0, 'Gefuehl': 0, 'Geist': 0, 'Gesellschaft': 0, 'Koerper': 0, 'Menge': 0, 'natPhaenomen': 0, 'Ort': 0, 'Pertonym': 0, 'Perzeption': 0, 'privativ': 0, 'Relation': 0, 'Substanz': 0, 'Verhalten': 0, 'Zeit': 0, 'Artefakt': 0, 'Attribut': 0, 'Besitz': 0, 'Form': 0, 'Geschehen': 0, 'Gruppe': 0, 'Kognition': 0, 'Kommunikation': 0, 'Mensch': 0, 'Motiv': 0, 'Nahrung': 0, 'natGegenstand': 0, 'Pflanze': 0, 'Tier': 0, 'Tops': 0, 'Koerperfunktion': 0, 'Konkurrenz': 0, 'Kontakt': 0, 'Lokation': 0, 'Schoepfung': 0, 'Veraenderung': 0, 'Verbrauch': 0}

""" To prevent unnecessary parsing, use formations already found in GermaNet """
GN_PREF_FORMATIONS = ['Blitzgerät', 'Blitzkarriere', 'Blitzkrieg', 'Blitzkurs', 'Blitzlampe', 'Blitzröhre', 'Blitzschach', 'Blitzschlag', 'Bombenabwurf', 'Bombenalarm', 'Bombendrohung', 'Bombenexplosion', 'Bombenleger', 'Bombennacht', 'Bombenopfer', 'Bombenschacht', 'Bombenschaden', 'Bombensplitter', 'Bombenteppich', 'Bombentest', 'Bombentrichter', 'Glanzente', 'Glanzleistung', 'Glanzlicht', 'Glanzpunkt', 'Glanzrolle', 'Glanzstoff', 'Glanzstück', 'Glanzzeit', 'Jahrhundertfeier', 'Jahrhunderthälfte', 'Jahrhunderthochwasser', 'Jahrhundertsommer', 'Jahrhundertwechsel', 'Qualitätsbewusstsein', 'Qualitätskriterium', 'Qualitätsprüfung', 'Qualitätsstandard', 'Qualitätsverbesserung', 'Qualitätswein', 'Qualitätszuwachs', 'Schweineblut', 'Schweinebraten', 'Schweinefleisch', 'Schweinehaltung', 'Schweinehirte', 'Schweinepest', 'Spitzenfunktionär', 'Spitzenkandidatin', 'Spitzenkoch', 'Spitzenläufer', 'Spitzenleistung', 'Spitzenplatz', 'Spitzenreiter', 'Spitzenspiel', 'Spitzensportler', 'Spitzenverband', 'Spitzenverdiener', 'Spitzenverein', 'Spitzenwert', 'Traummädchen', 'Traumwelt']
GN_SUFF_FORMATIONS = ['Börsenguru', 'Burgunderkönig', 'Bürohengst', 'Dänenkönig', 'Donnergott', 'Dreikönig', 'Feenkönig', 'Feuergott', 'Froschkönig', 'Gegenkönig', 'Gegenpapst', 'Gotenkönig', 'Gottkönig', 'Großkönig', 'Hausschwein', 'Herzkönig', 'Himmelsgott', 'Hochkönig', 'Hunnenkönig', 'Kleinkönig', 'Langobardenkönig', 'Liebesgott', 'Märchenkönig', 'Marienikone', 'Meeresgott', 'Moralapostel', 'Normannenkönig', 'Perserkönig', 'Preußenkönig', 'Priesterkönig', 'Rattenkönig', 'Schlagbolzen', 'Schöpfergott', 'Schützenkönig', 'Schwedenkönig', 'Slawenapostel', 'Sonnenkönig', 'Stammapostel', 'Torschützenkönig', 'Unterkönig', 'Vizekönig', 'Vogelkönig', 'Wachtelkönig', 'Warzenschwein', 'Westgotenkönig', 'Wettergott', 'Wildschwein', 'Winterkönig', 'Zaunkönig', 'Zuchthengst', 'Zwergenkönig']


class FeatureExtractor:
    """ FeatureExtractor Class

        Returns: Files with feature vectors

        Example: PREF = FeatureExtractor('Prefixoids', DATA_RESSOURCES_PATH + PREFIXOID_DICTIONARY)

    """

    def __init__(self, string, similar_words_dict):
        print('=' * 40)
        print(Style.BOLD + "Running FeatureExtractor on:" + Style.END, string)
        print('-' * 40)
        print('Initializing dictionary...')
        self.fasttext_similar_words_dict = fr.read_dict_from_file(similar_words_dict)

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
    PREF = FeatureExtractor('Prefixoids', config.get('FastTextSimilarities', 'PrefixoidDictionary'))
    pref_inventory_list = fr.read_file_to_list(config.get('FileSettings', 'FinalPrefixoidFile'))
    y_pref_dict = dc.create_affixoid_dictionary(config.get('FileSettings', 'FinalPrefixoidFile'), 'Y')
    n_pref_dict = dc.create_affixoid_dictionary(config.get('FileSettings', 'FinalPrefixoidFile'), 'N')
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

    f2_pref_formations = dc.create_frequency_dictionary(config.get('Frequencies', 'FrequencyPrefixoidFormations'))
    f3_pref_lemmas = dc.create_frequency_dictionary(config.get('Frequencies', 'FrequencyPrefixoidLemmas'))
    f4_pref_heads = dc.create_frequency_dictionary(config.get('Frequencies', 'FrequencyPrefixoidHeads'))
    f5_pref_vector_dict = dc.create_vector_dictionary(config.get('FastTextVectors', 'fastTextPrefixoidVectors'))
    f9_pref_polarity_dict = dc.create_polarity_dict(config.get('PsycholinguisticFeatures', 'SentimergePolarity'))
    f12_pref_affective_norms_dict = dc.create_vector_dictionary(config.get('AffectiveNorms', 'AffectiveNormsValues'))
    f15_pref_emolex_dict = dc.create_vector_dictionary(config.get('Emotion', 'EmoLex'), multiword=True)
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

        f0 = dc.extract_frequency(i[-3], y_pref_dict, True)  # y_pref_dict or n_pref_dict
        f1 = hf.transform_class_name_to_binary(i[-1])
        f2 = dc.extract_frequency(i[0], f2_pref_formations)
        f3 = dc.extract_frequency(hf.split_word_at_pipe(i[1])[0], f3_pref_lemmas)
        f4 = dc.extract_frequency(hf.split_word_at_pipe(i[1])[1], f4_pref_heads)
        f5 = PREF.calculate_cosine_similarity(i[0], hf.split_word_at_pipe(i[1])[0], f5_pref_vector_dict)  # split_word_at_pipe(i[1])[1] for SUFFIXOIDS
        f6 = PREF.search_germanet_supersenses(i[0], f5_pref_vector_dict)
        f7 = PREF.search_germanet_supersenses(hf.split_word_at_pipe(i[1])[0], f5_pref_vector_dict)
        f8 = PREF.search_germanet_supersenses(hf.split_word_at_pipe(i[1])[1], f5_pref_vector_dict)

        f9 = dc.extract_dictionary_values(i[0], f9_pref_polarity_dict)
        if f9 == 0:
            f9_similar_pol = PREF.return_single_word_from_fasttext(i[0], f9_pref_polarity_dict)
            if f9_similar_pol == 0:
                f9_similar_pol = PREF.return_similar_cosine_word(i[0], f5_pref_vector_dict, False, polarity_dict=f9_pref_polarity_dict)
                f9_similar_pol_value = dc.extract_dictionary_values(f9_similar_pol, f9_pref_polarity_dict)
            else:
                f9_similar_pol_value = dc.extract_dictionary_values(f9_similar_pol, f9_pref_polarity_dict)
            f9 = f9_similar_pol_value

        f10 = dc.extract_dictionary_values(hf.split_word_at_pipe(i[1])[0], f9_pref_polarity_dict)
        if f10 == 0:
            f10_similar_pol = PREF.return_single_word_from_fasttext(hf.split_word_at_pipe(i[1])[0], f9_pref_polarity_dict)
            if f10_similar_pol == 0:
                f10_similar_pol = PREF.return_similar_cosine_word(hf.split_word_at_pipe(i[1])[0], f5_pref_vector_dict, False, polarity_dict=f9_pref_polarity_dict)
                f10_similar_pol_value = dc.extract_dictionary_values(f10_similar_pol, f9_pref_polarity_dict)
            else:
                f10_similar_pol_value = dc.extract_dictionary_values(f10_similar_pol, f9_pref_polarity_dict)
            f10 = f10_similar_pol_value

        f11 = dc.extract_dictionary_values(hf.split_word_at_pipe(i[1])[1], f9_pref_polarity_dict)
        if f11 == 0:
            f11_similar_pol = PREF.return_single_word_from_fasttext(hf.split_word_at_pipe(i[1])[1], f9_pref_polarity_dict)
            if f11_similar_pol == 0:
                f11_similar_pol = PREF.return_similar_cosine_word(hf.split_word_at_pipe(i[1])[1], f5_pref_vector_dict, False, polarity_dict=f9_pref_polarity_dict)
                f11_similar_pol_value = dc.extract_dictionary_values(f11_similar_pol, f9_pref_polarity_dict)
            else:
                f11_similar_pol_value = dc.extract_dictionary_values(f11_similar_pol, f9_pref_polarity_dict)
            f11 = f11_similar_pol_value

        f12 = dc.extract_dictionary_values(i[0], f12_pref_affective_norms_dict)
        if f12 == 0:
            f12_similar_pol = PREF.return_single_word_from_fasttext(i[0], f12_pref_affective_norms_dict)
            if f12_similar_pol == 0:
                f12_similar_pol = PREF.return_similar_cosine_word(i[0], f5_pref_vector_dict, False, polarity_dict=f12_pref_affective_norms_dict)
                f12_similar_pol_value = dc.extract_dictionary_values(f12_similar_pol, f12_pref_affective_norms_dict)
            else:
                f12_similar_pol_value = dc.extract_dictionary_values(f12_similar_pol, f12_pref_affective_norms_dict)
            f12 = f12_similar_pol_value

        f13 = dc.extract_dictionary_values(hf.split_word_at_pipe(i[1])[0], f12_pref_affective_norms_dict)
        if f13 == 0:
            f13_similar_pol = PREF.return_single_word_from_fasttext(hf.split_word_at_pipe(i[1])[0], f12_pref_affective_norms_dict)
            if f13_similar_pol == 0:
                f13_similar_pol = PREF.return_similar_cosine_word(hf.split_word_at_pipe(i[1])[0], f5_pref_vector_dict, False, polarity_dict=f12_pref_affective_norms_dict)
                f13_similar_pol_value = dc.extract_dictionary_values(f13_similar_pol, f12_pref_affective_norms_dict)
            else:
                f13_similar_pol_value = dc.extract_dictionary_values(f13_similar_pol, f12_pref_affective_norms_dict)
            f13 = f13_similar_pol_value

        f14 = dc.extract_dictionary_values(hf.split_word_at_pipe(i[1])[1], f12_pref_affective_norms_dict)
        if f14 == 0:
            f14_similar_pol = PREF.return_single_word_from_fasttext(hf.split_word_at_pipe(i[1])[1], f12_pref_affective_norms_dict)
            if f14_similar_pol == 0:
                f14_similar_pol = PREF.return_similar_cosine_word(hf.split_word_at_pipe(i[1])[1], f5_pref_vector_dict, False, polarity_dict=f12_pref_affective_norms_dict)
                f14_similar_pol_value = dc.extract_dictionary_values(f14_similar_pol, f12_pref_affective_norms_dict)
            else:
                f14_similar_pol_value = dc.extract_dictionary_values(f14_similar_pol, f12_pref_affective_norms_dict)
            f14 = f14_similar_pol_value

        f15 = dc.extract_dictionary_values(i[0], f15_pref_emolex_dict)
        if f15 == 0:
            f15_similar_pol = PREF.return_single_word_from_fasttext(i[0], f15_pref_emolex_dict)
            if f15_similar_pol == 0:
                f15_similar_pol = PREF.return_similar_cosine_word(i[0], f5_pref_vector_dict, False, polarity_dict=f15_pref_emolex_dict)
                f15_similar_pol_value = dc.extract_dictionary_values(f15_similar_pol, f15_pref_emolex_dict)
            else:
                f15_similar_pol_value = dc.extract_dictionary_values(f15_similar_pol, f15_pref_emolex_dict)
            f15 = f15_similar_pol_value

        f16 = dc.extract_dictionary_values(hf.split_word_at_pipe(i[1])[0], f15_pref_emolex_dict)
        if f16 == 0:
            f16_similar_pol = PREF.return_single_word_from_fasttext(hf.split_word_at_pipe(i[1])[0], f15_pref_emolex_dict)
            if f16_similar_pol == 0:
                f16_similar_pol = PREF.return_similar_cosine_word(hf.split_word_at_pipe(i[1])[0], f5_pref_vector_dict, False, polarity_dict=f15_pref_emolex_dict)
                f16_similar_pol_value = dc.extract_dictionary_values(f16_similar_pol, f15_pref_emolex_dict)
            else:
                f16_similar_pol_value = dc.extract_dictionary_values(f16_similar_pol, f15_pref_emolex_dict)
            f16 = f16_similar_pol_value

        f17 = dc.extract_dictionary_values(hf.split_word_at_pipe(i[1])[1], f15_pref_emolex_dict)
        if f17 == 0:
            f17_similar_pol = PREF.return_single_word_from_fasttext(hf.split_word_at_pipe(i[1])[1], f15_pref_emolex_dict)
            if f17_similar_pol == 0:
                f17_similar_pol = PREF.return_similar_cosine_word(hf.split_word_at_pipe(i[1])[1], f5_pref_vector_dict, False, polarity_dict=f15_pref_emolex_dict)
                f17_similar_pol_value = dc.extract_dictionary_values(f17_similar_pol, f15_pref_emolex_dict)
            else:
                f17_similar_pol_value = dc.extract_dictionary_values(f17_similar_pol, f15_pref_emolex_dict)
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
    fw.write_list_to_file(f0_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f0_pref.txt')  # DONE
    fw.write_list_to_file(f1_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f1_pref.txt')  # DONE
    fw.write_list_to_file(f2_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f2_pref.txt')  # DONE
    fw.write_list_to_file(f3_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f3_pref.txt')  # DONE
    fw.write_list_to_file(f4_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f4_pref.txt')  # DONE
    fw.write_list_to_file(f5_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f5_pref.txt')  # DONE
    fw.write_list_to_file(f6_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f6_pref.txt')  # DONE
    fw.write_list_to_file(f7_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f7_pref.txt')  # DONE
    fw.write_list_to_file(f8_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f8_pref.txt')  # DONE
    fw.write_list_to_file(f9_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f9_pref.txt')  # DONE
    fw.write_list_to_file(f10_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f10_pref.txt')  # DONE
    fw.write_list_to_file(f11_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f11_pref.txt')  # DONE
    fw.write_list_to_file(f12_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f12_pref.txt')  # DONE
    fw.write_list_to_file(f13_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f13_pref.txt')  # DONE
    fw.write_list_to_file(f14_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f14_pref.txt')  # DONE
    fw.write_list_to_file(f15_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f15_pref.txt')  # DONE
    fw.write_list_to_file(f16_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f16_pref.txt')  # DONE
    fw.write_list_to_file(f17_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f17_pref.txt')  # DONE

    """ PMI Scores """
    f18_pref_pmi_dict = dc.extract_pmi_values(config.get('PmiLexicon', 'PmiScores'), f18_pref_splitwords_dict, config.get('PmiLexicon', 'PmiOutput') + 'pref_PMI_scores.txt')
    f18_pref_list = []  # PMI Scores for first and second part of word

    """ Second loop over inventory (after collecting splitwords) """
    for i in pref_inventory_list:
        f18 = dc.extract_dictionary_values(i[1], f18_pref_pmi_dict)
        f18_pref_list.append(f18)

    # print(f18_pref_list)
    print(len(f18_pref_list))

    fw.write_list_to_file(f18_pref_list, config.get('PathSettings', 'DataFeaturesPath') + 'f18_pref.txt')

    """
        SUFFIXOIDS
    """
    SUFF = FeatureExtractor('Suffixoids', config.get('FastTextSimilarities', 'SuffixoidDictionary'))
    suff_inventory_list = fr.read_file_to_list(config.get('FileSettings', 'FinalSuffixoidFile'))
    y_suff_dict = dc.create_affixoid_dictionary(config.get('FileSettings', 'FinalSuffixoidFile'), 'Y')
    n_suff_dict = dc.create_affixoid_dictionary(config.get('FileSettings', 'FinalSuffixoidFile'), 'N')
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

    f2_suff_formations = dc.create_frequency_dictionary(config.get('Frequencies', 'FrequencySuffixoidFormations'))
    f3_suff_lemmas = dc.create_frequency_dictionary(config.get('Frequencies', 'FrequencySuffixoidModifiers'))
    f4_suff_heads = dc.create_frequency_dictionary(config.get('Frequencies', 'FrequencySuffixoidLemmas'))
    f5_suff_vector_dict = dc.create_vector_dictionary(config.get('FastTextVectors', 'fastTextSuffixoidVectors'))
    f9_suff_polarity_dict = dc.create_polarity_dict(config.get('PsycholinguisticFeatures', 'SentimergePolarity'))
    f12_suff_affective_norms_dict = dc.create_vector_dictionary(config.get('AffectiveNorms', 'AffectiveNormsValues'))
    f15_suff_emolex_dict = dc.create_vector_dictionary(config.get('Emotion', 'EmoLex'), multiword=True)
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

        f0 = dc.extract_frequency(i[-3], y_suff_dict, True)  # y_suff_dict or n_suff_dict
        f1 = hf.transform_class_name_to_binary(i[-1])
        f2 = dc.extract_frequency(i[0], f2_suff_formations)
        f3 = dc.extract_frequency(hf.split_word_at_pipe(i[1])[0], f3_suff_lemmas)
        f4 = dc.extract_frequency(hf.split_word_at_pipe(i[1])[1], f4_suff_heads)
        f5 = SUFF.calculate_cosine_similarity(i[0], hf.split_word_at_pipe(i[1])[1], f5_suff_vector_dict)  # split_word_at_pipe(i[1])[0] for PREFIXOIDS
        f6 = SUFF.search_germanet_supersenses(i[0], f5_suff_vector_dict)
        f7 = SUFF.search_germanet_supersenses(hf.split_word_at_pipe(i[1])[0], f5_suff_vector_dict)
        f8 = SUFF.search_germanet_supersenses(hf.split_word_at_pipe(i[1])[1], f5_suff_vector_dict)

        f9 = dc.extract_dictionary_values(i[0], f9_suff_polarity_dict)
        if f9 == 0:
            f9_similar_pol = SUFF.return_single_word_from_fasttext(i[0], f9_suff_polarity_dict)
            if f9_similar_pol == 0:
                f9_similar_pol = SUFF.return_similar_cosine_word(i[0], f5_suff_vector_dict, False, polarity_dict=f9_suff_polarity_dict)
                f9_similar_pol_value = dc.extract_dictionary_values(f9_similar_pol, f9_suff_polarity_dict)
            else:
                f9_similar_pol_value = dc.extract_dictionary_values(f9_similar_pol, f9_suff_polarity_dict)
            f9 = f9_similar_pol_value

        f10 = dc.extract_dictionary_values(hf.split_word_at_pipe(i[1])[0], f9_suff_polarity_dict)
        if f10 == 0:
            f10_similar_pol = SUFF.return_single_word_from_fasttext(hf.split_word_at_pipe(i[1])[0], f9_suff_polarity_dict)
            if f10_similar_pol == 0:
                f10_similar_pol = SUFF.return_similar_cosine_word(hf.split_word_at_pipe(i[1])[0], f5_suff_vector_dict, False, polarity_dict=f9_suff_polarity_dict)
                f10_similar_pol_value = dc.extract_dictionary_values(f10_similar_pol, f9_suff_polarity_dict)
            else:
                f10_similar_pol_value = dc.extract_dictionary_values(f10_similar_pol, f9_suff_polarity_dict)
            f10 = f10_similar_pol_value

        f11 = dc.extract_dictionary_values(hf.split_word_at_pipe(i[1])[1], f9_suff_polarity_dict)
        if f11 == 0:
            f11_similar_pol = SUFF.return_single_word_from_fasttext(hf.split_word_at_pipe(i[1])[1], f9_suff_polarity_dict)
            if f11_similar_pol == 0:
                f11_similar_pol = SUFF.return_similar_cosine_word(hf.split_word_at_pipe(i[1])[1], f5_suff_vector_dict, False, polarity_dict=f9_suff_polarity_dict)
                f11_similar_pol_value = dc.extract_dictionary_values(f11_similar_pol, f9_suff_polarity_dict)
            else:
                f11_similar_pol_value = dc.extract_dictionary_values(f11_similar_pol, f9_suff_polarity_dict)
            f11 = f11_similar_pol_value

        f12 = dc.extract_dictionary_values(i[0], f12_suff_affective_norms_dict)
        if f12 == 0:
            f12_similar_pol = SUFF.return_single_word_from_fasttext(i[0], f12_suff_affective_norms_dict)
            if f12_similar_pol == 0:
                f12_similar_pol = SUFF.return_similar_cosine_word(i[0], f5_suff_vector_dict, False, polarity_dict=f12_suff_affective_norms_dict)
                f12_similar_pol_value = dc.extract_dictionary_values(f12_similar_pol, f12_suff_affective_norms_dict)
            else:
                f12_similar_pol_value = dc.extract_dictionary_values(f12_similar_pol, f12_suff_affective_norms_dict)
            f12 = f12_similar_pol_value

        f13 = dc.extract_dictionary_values(hf.split_word_at_pipe(i[1])[0], f12_suff_affective_norms_dict)
        if f13 == 0:
            f13_similar_pol = SUFF.return_single_word_from_fasttext(hf.split_word_at_pipe(i[1])[0], f12_suff_affective_norms_dict)
            if f13_similar_pol == 0:
                f13_similar_pol = SUFF.return_similar_cosine_word(hf.split_word_at_pipe(i[1])[0], f5_suff_vector_dict, False, polarity_dict=f12_suff_affective_norms_dict)
                f13_similar_pol_value = dc.extract_dictionary_values(f13_similar_pol, f12_suff_affective_norms_dict)
            else:
                f13_similar_pol_value = dc.extract_dictionary_values(f13_similar_pol, f12_suff_affective_norms_dict)
            f13 = f13_similar_pol_value

        f14 = dc.extract_dictionary_values(hf.split_word_at_pipe(i[1])[1], f12_suff_affective_norms_dict)
        if f14 == 0:
            f14_similar_pol = SUFF.return_single_word_from_fasttext(hf.split_word_at_pipe(i[1])[1], f12_suff_affective_norms_dict)
            if f14_similar_pol == 0:
                f14_similar_pol = SUFF.return_similar_cosine_word(hf.split_word_at_pipe(i[1])[1], f5_suff_vector_dict, False, polarity_dict=f12_suff_affective_norms_dict)
                f14_similar_pol_value = dc.extract_dictionary_values(f14_similar_pol, f12_suff_affective_norms_dict)
            else:
                f14_similar_pol_value = dc.extract_dictionary_values(f14_similar_pol, f12_suff_affective_norms_dict)
            f14 = f14_similar_pol_value

        f15 = dc.extract_dictionary_values(i[0], f15_suff_emolex_dict)
        if f15 == 0:
            f15_similar_pol = SUFF.return_single_word_from_fasttext(i[0], f15_suff_emolex_dict)
            if f15_similar_pol == 0:
                f15_similar_pol = SUFF.return_similar_cosine_word(i[0], f5_suff_vector_dict, False, polarity_dict=f15_suff_emolex_dict)
                f15_similar_pol_value = dc.extract_dictionary_values(f15_similar_pol, f15_suff_emolex_dict)
            else:
                f15_similar_pol_value = dc.extract_dictionary_values(f15_similar_pol, f15_suff_emolex_dict)
            f15 = f15_similar_pol_value

        f16 = dc.extract_dictionary_values(hf.split_word_at_pipe(i[1])[0], f15_suff_emolex_dict)
        if f16 == 0:
            f16_similar_pol = SUFF.return_single_word_from_fasttext(hf.split_word_at_pipe(i[1])[0], f15_suff_emolex_dict)
            if f16_similar_pol == 0:
                f16_similar_pol = SUFF.return_similar_cosine_word(hf.split_word_at_pipe(i[1])[0], f5_suff_vector_dict, False, polarity_dict=f15_suff_emolex_dict)
                f16_similar_pol_value = dc.extract_dictionary_values(f16_similar_pol, f15_suff_emolex_dict)
            else:
                f16_similar_pol_value = dc.extract_dictionary_values(f16_similar_pol, f15_suff_emolex_dict)
            f16 = f16_similar_pol_value

        f17 = dc.extract_dictionary_values(hf.split_word_at_pipe(i[1])[1], f15_suff_emolex_dict)
        if f17 == 0:
            f17_similar_pol = SUFF.return_single_word_from_fasttext(hf.split_word_at_pipe(i[1])[1], f15_suff_emolex_dict)
            if f17_similar_pol == 0:
                f17_similar_pol = SUFF.return_similar_cosine_word(hf.split_word_at_pipe(i[1])[1], f5_suff_vector_dict, False, polarity_dict=f15_suff_emolex_dict)
                f17_similar_pol_value = dc.extract_dictionary_values(f17_similar_pol, f15_suff_emolex_dict)
            else:
                f17_similar_pol_value = dc.extract_dictionary_values(f17_similar_pol, f15_suff_emolex_dict)
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
    fw.write_list_to_file(f0_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f0_suff.txt')  # DONE
    fw.write_list_to_file(f1_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f1_suff.txt')  # DONE
    fw.write_list_to_file(f2_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f2_suff.txt')  # DONE
    fw.write_list_to_file(f3_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f3_suff.txt')  # DONE
    fw.write_list_to_file(f4_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f4_suff.txt')  # DONE
    fw.write_list_to_file(f5_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f5_suff.txt')  # DONE
    fw.write_list_to_file(f6_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f6_suff.txt')  # DONE
    fw.write_list_to_file(f7_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f7_suff.txt')  # DONE
    fw.write_list_to_file(f8_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f8_suff.txt')  # DONE
    fw.write_list_to_file(f9_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f9_suff.txt')  # DONE
    fw.write_list_to_file(f10_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f10_suff.txt')  # DONE
    fw.write_list_to_file(f11_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f11_suff.txt')  # DONE
    fw.write_list_to_file(f12_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f12_suff.txt')  # DONE
    fw.write_list_to_file(f13_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f13_suff.txt')  # DONE
    fw.write_list_to_file(f14_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f14_suff.txt')  # DONE
    fw.write_list_to_file(f15_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f15_suff.txt')  # DONE
    fw.write_list_to_file(f16_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f16_suff.txt')  # DONE
    fw.write_list_to_file(f17_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f17_suff.txt')  # DONE

    """ PMI Scores """
    f18_suff_pmi_dict = dc.extract_pmi_values(config.get('PmiLexicon', 'PmiScores'), f18_suff_splitwords_dict, config.get('PmiLexicon', 'PmiOutput') + 'suff_PMI_scores.txt')
    f18_suff_list = []  # PMI Scores for first and second part of word

    """ Second loop over inventory (after collecting splitwords) """
    for i in suff_inventory_list:
        f18 = dc.extract_dictionary_values(i[1], f18_suff_pmi_dict)
        f18_suff_list.append(f18)

    # print(f18_suff_list)
    print(len(f18_suff_list))

    fw.write_list_to_file(f18_suff_list, config.get('PathSettings', 'DataFeaturesPath') + 'f18_suff.txt')
