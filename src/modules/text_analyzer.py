#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
text_analyzer

Short description:
This module analyzes text

License: MIT License
Version: 1.0

"""

import configparser
import requests
import sys
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import load
from nltk.tokenize import RegexpTokenizer
from pygermanet import load_germanet
from modules import file_reader as fr

########################
# GLOBAL FILE SETTINGS
########################
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('../config.ini')

########################
# GermaNet & WordNet
########################
try:
    ger = load_germanet()
except:
    print('Error! Please start mongodb on GermaNet xml files: mongod --dbpath ./mongodb or refer to README.md')
    sys.exit()

# Tokenizer
sent_tok = load('tokenizers/punkt/german.pickle')
word_tok = RegexpTokenizer(r'\w+')

# Filter stopwords
german_stopwords = stopwords.words('german')
german_stopwords.extend(('dass', 'bzw', 'p', 'http', '0', '1', '2', '3', '4'))
stop_words = set(german_stopwords)

# Corpora
corpname = 'sdewac2'


def search_duden_frequency(words_inventory):
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


# def is_in_germanet_fast(word):
#     """ A slightly faster version that parses GermaNet for a word and returns a boolean if the word is found """
#
#     if GN_ROOT.xpath('.//orthForm[text()="'+word+'"]') is not None:
#         return True
#     else:
#         return False

dict_n = fr.read_dict_from_file('../' + config.get('PathSettings', 'DataWsdPath') + corpname + '/pref_n.txt')
dict_y = fr.read_dict_from_file('../' + config.get('PathSettings', 'DataWsdPath') + corpname + '/pref_y.txt')

# dict_n = fr.read_dict_from_file('../' + config.get('PathSettings', 'DataWsdPath') + corpname + '/suff_n.txt')
# dict_y = fr.read_dict_from_file('../' + config.get('PathSettings', 'DataWsdPath') + corpname + '/suff_y.txt')


for i in dict_n.keys():
    print(i)
    text_n = str(dict_n[i])
    text_y = str(dict_y[i])

    # print(text_n)
    # print('---')
    # print(text_y)

    words_n = [w for w in word_tok.tokenize(text_n) if w.lower() not in stop_words]
    words_y = [w for w in word_tok.tokenize(text_y) if w.lower() not in stop_words]

    words_n_lemmatized = [ger.lemmatise(w) for w in words_n]
    words_y_lemmatized = [ger.lemmatise(w) for w in words_y]

    fdistn = FreqDist(words_n)
    fdisty = FreqDist(words_y)

    most_common_n = fdistn.most_common(50)
    most_common_y = fdisty.most_common(50)

    print(fdistn[i])
    print(fdisty[i])

    hapax_n = FreqDist.hapaxes(fdistn)
    hapax_y = FreqDist.hapaxes(fdisty)

    list_n = [n[0] for n in most_common_n if n[0][0].isupper() and len(n[0]) > 1]
    list_y = [n[0] for n in most_common_y if n[0][0].isupper() and len(n[0]) > 1]

    # print(list_n)
    # print(list_y)

    print(set(list_n).difference(list_y))
    print(set(list_y).difference(list_n))

    # for m in most_common_n:
    #     if m[0] in set(hapax_n):
    #         print(m[0])

    # print(set(hapax_n))
    # print(set(hapax_n).difference(hapax_y))

    print("--==--")

