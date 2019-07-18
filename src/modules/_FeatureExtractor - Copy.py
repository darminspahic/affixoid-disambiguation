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
import duden
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import matplotlib.pyplot as plt
import ast
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

"""Frequencies"""
FREQUENCY_PREFIXOID_FORMATIONS = 'lemma_frequencies_prefixoid_formations.csv'
FREQUENCY_SUFFIXOID_FORMATIONS = 'lemma_frequencies_suffixoid_formations.csv'
FREQUENCY_PREFIXOID_LEMMAS = 'prefixoid_lemmas_freqs.csv'
FREQUENCY_SUFFIXOID_LEMMAS = 'suffixoid_lemmas_freqs.csv'
FREQUENCY_PREFIXOID_HEADS = 'lemma_frequencies_unique_heads_of_prefixoid_formations.csv'
FREQUENCY_SUFFIXOID_MODIFIERS = 'modifiers_of_suffixoids_lemmas_freqs.csv'

"""fastText vectors"""
FAST_TEXT_PREFIXOID_VECTORS = 'fastText/prefixoid-fastText-vectors.txt'
FAST_TEXT_SUFFIXOID_VECTORS = 'fastText/suffixoid-fastText-vectors.txt'

"""Polarity"""
SENTIMERGE_POLARITY = 'SentiMerge/sentimerge.txt'

"""Psycholinguistic features; Affective norms"""
AFFECTIVE_NORMS = 'AffectiveNorms/ratings_lrec16_koeper_ssiw.txt'

"""Emolex"""
EMOLEX = 'EmoLex/NRC-Emotion-Lexicon-v0.92-DE-sorted.csv'

################
# GermaNet
################
# TREE = etree.parse(DATA_RESSOURCES_PATH+'GermaNet/GN_full.xml')
# GN_ROOT = TREE.getroot()
# GN_WORDS = GN_ROOT.findall('.//orthForm')

"""GermaNet Supersenses"""
GN_SUPERSENSES = {'Allgemein': 0, 'Bewegung': 0, 'Gefuehl': 0, 'Geist': 0, 'Gesellschaft': 0, 'Koerper': 0, 'Menge': 0, 'natPhaenomen': 0, 'Ort': 0, 'Pertonym': 0, 'Perzeption': 0, 'privativ': 0, 'Relation': 0, 'Substanz': 0, 'Verhalten': 0, 'Zeit': 0, 'Artefakt': 0, 'Attribut': 0, 'Besitz': 0, 'Form': 0, 'Geschehen': 0, 'Gruppe': 0, 'Kognition': 0, 'Kommunikation': 0, 'Mensch': 0, 'Motiv': 0, 'Nahrung': 0, 'natGegenstand': 0, 'Pflanze': 0, 'Tier': 0, 'Tops': 0, 'Koerperfunktion': 0, 'Konkurrenz': 0, 'Kontakt': 0, 'Lokation': 0, 'Schoepfung': 0, 'Veraenderung': 0, 'Verbrauch': 0}

"""To prevent unnecessary parsing, use words already found"""
GN_PREF_FORMATIONS = ['Blitzgerät', 'Blitzkarriere', 'Blitzkrieg', 'Blitzkurs', 'Blitzlampe', 'Blitzröhre', 'Blitzschach', 'Blitzschlag', 'Bombenabwurf', 'Bombenalarm', 'Bombendrohung', 'Bombenexplosion', 'Bombenleger', 'Bombennacht', 'Bombenopfer', 'Bombenschacht', 'Bombenschaden', 'Bombensplitter', 'Bombenteppich', 'Bombentest', 'Bombentrichter', 'Glanzente', 'Glanzleistung', 'Glanzlicht', 'Glanzpunkt', 'Glanzrolle', 'Glanzstoff', 'Glanzstück', 'Glanzzeit', 'Jahrhundertfeier', 'Jahrhunderthälfte', 'Jahrhunderthochwasser', 'Jahrhundertsommer', 'Jahrhundertwechsel', 'Qualitätsbewusstsein', 'Qualitätskriterium', 'Qualitätsprüfung', 'Qualitätsstandard', 'Qualitätsverbesserung', 'Qualitätswein', 'Qualitätszuwachs', 'Schweineblut', 'Schweinebraten', 'Schweinefleisch', 'Schweinehaltung', 'Schweinehirte', 'Schweinepest', 'Spitzenfunktionär', 'Spitzenkandidatin', 'Spitzenkoch', 'Spitzenläufer', 'Spitzenleistung', 'Spitzenplatz', 'Spitzenreiter', 'Spitzenspiel', 'Spitzensportler', 'Spitzenverband', 'Spitzenverdiener', 'Spitzenverein', 'Spitzenwert', 'Traummädchen', 'Traumwelt']
GN_SUFF_FORMATIONS = ['Börsenguru', 'Burgunderkönig', 'Bürohengst', 'Dänenkönig', 'Donnergott', 'Dreikönig', 'Feenkönig', 'Feuergott', 'Froschkönig', 'Gegenkönig', 'Gegenpapst', 'Gotenkönig', 'Gottkönig', 'Großkönig', 'Hausschwein', 'Herzkönig', 'Himmelsgott', 'Hochkönig', 'Hunnenkönig', 'Kleinkönig', 'Langobardenkönig', 'Liebesgott', 'Märchenkönig', 'Marienikone', 'Meeresgott', 'Moralapostel', 'Normannenkönig', 'Perserkönig', 'Preußenkönig', 'Priesterkönig', 'Rattenkönig', 'Schlagbolzen', 'Schöpfergott', 'Schützenkönig', 'Schwedenkönig', 'Slawenapostel', 'Sonnenkönig', 'Stammapostel', 'Torschützenkönig', 'Unterkönig', 'Vizekönig', 'Vogelkönig', 'Wachtelkönig', 'Warzenschwein', 'Westgotenkönig', 'Wettergott', 'Wildschwein', 'Winterkönig', 'Zaunkönig', 'Zuchthengst', 'Zwergenkönig']

"""Affixoid dictionary with fastText similarities; sorted"""
AFFIXOID_DICTIONARY = 'fastText/affixoid_dict_fasttext_similarites.txt'

"""Empty words dictionary for collecting various data"""
EMPTY_WORDS_DICTIONARY = 'all_words_dict.txt'


class FeatureExtractor:
    """ FeatureExtractor Class

        Returns: Files with feature vectors

        Example: CLSF = FeatureExtractor()

    """

    def __init__(self, string):
        print('=' * 40)
        print(Style.BOLD + "Running FeatureExtractor on:", string + Style.END)
        print('-' * 40)

        try:
            self.fasttext_similar_words_dict = self.read_dict_from_file(DATA_RESSOURCES_PATH + AFFIXOID_DICTIONARY)
            self.empty_words_dict = self.read_dict_from_file(DATA_FINAL_PATH + EMPTY_WORDS_DICTIONARY)

        except FileNotFoundError:
            print('Please set correct paths for data.')

        cand = {'Bilderbuch': [], 'Blitz': [], 'Bombe': [], 'Glanz': [], 'Heide': [], 'Jahrhundert': [], 'Qualität': [], 'Schwein': [], 'Spitze': [], 'Traum': [],
                'Apostel': [], 'Bolzen': [], 'Dreck': [], 'Gott': [], 'Guru': [], 'Hengst': [], 'Ikone': [], 'König': [], 'Papst': []}

        counter = 0
        c2 = cand.copy()
        for key in cand:
            counter += 1
            print()

            print('Line:', str(counter) + ' ===============================')
            # if counter == 50:
            #     break
            try:
                w = duden.get(key)
                bedeutungen = w.meaning_overview
                synonyme = w.synonyms
                c2.update({w.name: [{'Bedeutung': bedeutungen}, {'Synonyme': synonyme}]})
                # print('Title:', w.title)
                # print('Name:', w.name)
                # print('Bedeutung:', w.meaning_overview)
                # print('Synonyme:', w.synonyms)
                print('====================')
            except:
                pass

        print(c2)

    def write_dict_to_file(self, dictionary, output_file):
        """ Helper function to write a dictionary as string to a file. Import via ast module """

        with open(output_file, 'wb') as data:
            data.write(str(dictionary).encode('utf-8'))

        print(Style.BOLD + 'Dictionary written to:' + Style.END, output_file)

    def read_dict_from_file(self, dictionary_file):
        with open(dictionary_file, "rb") as data:
            dictionary = ast.literal_eval(data.read().decode('utf-8'))

        return dictionary

    def search_duden_frequency(self, words_inventory):
        if type(words_inventory) != list:
            words_inventory = words_inventory.split()

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

        words_inventory = replace_umlauts(words_inventory)
        frequency_list = []

        for w in words_inventory:
            words = duden.get(w)
            if words:
                try:
                    frequency_list.append(words.frequency)
                except AttributeError:
                    frequency_list.append(0)
            else:
                first_word = get_first_result(w)
                words = duden.get(first_word)
                try:
                    frequency_list.append(words.frequency)
                except AttributeError:
                    frequency_list.append(0)

        return frequency_list


class Style:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    UNDERLINE = '\033[4m'
    BOLD = '\033[1m'
    END = '\033[0m'


if __name__ == "__main__":
    PREF = FeatureExtractor('Duden')

    # print(CLSF.search_duden_frequency(['Arbeit', 'Urlaub']))
    #
    # w = duden.get('Arbeit')
    #
    # print(w.title)
    # print(w.name)
    # print(w.meaning_overview)
    # print(w.synonyms)



    """
    > w.title
    'Barmherzigkeit, die'
    
    > w.name
    'Barmherzigkeit'
    
    > w.article
    'die'
    
    > w.part_of_speech
    'Substantiv, feminin'
    
    > w.frequency
    2
    
    > w.usage
    'gehoben'
    
    > w.word_separation
    ['Barm', 'her', 'zig', 'keit']
    
    > w.meaning_overview
    'barmherziges Wesen, Verhalten'
    
    > w.synonyms
    '[Engels]güte, Milde, Nachsicht, Nachsichtigkeit; (gehoben) Herzensgüte, Mildtätigkeit, Seelengüte; (bildungssprachlich) Humanität, Indulgenz; (veraltend) Wohltätigkeit; (Religion) Gnade'
    
    > w.origin
    'mittelhochdeutsch barmherzekeit, barmherze, althochdeutsch armherzi, nach (kirchen)lateinisch misericordia'
    
    > w.compounds
    None
    """


