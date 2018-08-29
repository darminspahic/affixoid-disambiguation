#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
StatisticsExtractor

Short description:
This module extracts affixoid statistics from dlexdb, Wiktionary, GermaNet, SentiMerge and Duden

License: MIT License
Version: 1.0

"""

import configparser
import requests
import time
import duden

from modules import dictionaries as dc
from modules import file_reader as fr
from modules import statistics as st

from lxml import etree

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

# dlexdb API url
dlexdb_url = 'http://dlexdb.de/sr/dlexdb/kern/typ/list/?'

# wiktionary API url
wiktionary_url = 'https://de.wiktionary.org/w/api.php?'

sentimerge_dict = dc.create_polarity_dict(config.get('PsycholinguisticFeatures', 'SentimergePolarity'))
s = requests.Session()


def parse_dlexdb(word):
    """ This function parses dlexdb for a word and returns a frequency value if the word is found """
    headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
    params = dict(
        select='typ_freq_abs',
        list_eq=word,
        skip='',
        orderby='',
        top='',
    )

    try:
        resp = requests.get(url=dlexdb_url, headers=headers, params=params)
        data = resp.json()
    except:
        time.sleep(10)
        resp = requests.get(url=dlexdb_url, headers=headers, params=params)
        data = resp.json()

    if data['data'][0][0] is not None:
        print(word, data['data'][0][0])
        return data['data'][0][0]
    else:
        return 0


def parse_wictionary(word):
    """ This function parses Wiktionary for a word and returns a positive value value if the word is found """
    headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
    params = dict(
        action='query',
        titles=word,
        format='json'
    )

    try:
        resp = requests.get(url=wiktionary_url, headers=headers, params=params)
        data = resp.json()
    except:
        time.sleep(10)
        resp = requests.get(url=wiktionary_url, headers=headers, params=params)
        data = resp.json()

    if int(list(data['query']['pages'].keys())[0]) > -1:
        print(data['query']['pages'])
        return 1
    else:
        return 0


def is_in_germanet(word):
    """ This function parses GermaNet for a word and returns a positive value if the word is found """
    for item in GN_WORDS:
        if word == item.text or word == item.text.lower() or word == item.text.lower().capitalize():
            print(word)
            return 1
    return 0


def is_in_sentimerge(word):
    """ This function parses SentiMerge for a word and returns a positive value if the word is found """
    if word.lower() in sentimerge_dict.keys():
        print(word)
        return 1
    else:
        return 0


def is_in_duden(word):
    """ This function parses duden.de for a word and returns a positive value if the word is found """
    try:
        word_in_duden = duden.get(word)
    except:
        print('Connection attempt failed.')
        return False
    if word_in_duden:
        print(word)
        return 1
    else:
        return 0


if __name__ == "__main__":
    """
        PREFIXOIDS
    """
    pref_inventory_list = fr.read_file_to_list(config.get('FileSettings', 'FinalPrefixoidFile'))
    y_pref_dict_total = dc.create_affixoid_dictionary(config.get('FileSettings', 'FinalPrefixoidFile'), 'Y')
    n_pref_dict_total = dc.create_affixoid_dictionary(config.get('FileSettings', 'FinalPrefixoidFile'), 'N')
    y_pref_dictionary = dc.create_empty_dictionary(config.get('FileSettings', 'FinalPrefixoidFile'))
    n_pref_dictionary = dc.create_empty_dictionary(config.get('FileSettings', 'FinalPrefixoidFile'))
    print('Total:\t', len(pref_inventory_list))

    methods = [parse_dlexdb, parse_wictionary, is_in_germanet, is_in_sentimerge, is_in_duden]

    for m in methods:
        print('=' * 40)
        print(str(m))
        y_pref_dict = y_pref_dictionary.copy()
        n_pref_dict = n_pref_dictionary.copy()
        for k in y_pref_dict.keys():
            c_y = 0
            c_n = 0
            for p in pref_inventory_list:
                if k == p[-3]:
                    frequency = m(p[0])
                    if frequency > 0:
                        if p[-1] == 'Y':
                            c_y += 1
                            y_pref_dict.update({p[-3]: c_y})
                        if p[-1] == 'N':
                            c_n += 1
                            n_pref_dict.update({p[-3]: c_n})
                    else:
                        pass
                else:
                    pass

        print('Y:\t', y_pref_dict)
        print('N:\t', n_pref_dict)
        print()

    """
        SUFFIXOIDS
    """
    suff_inventory_list = fr.read_file_to_list(config.get('FileSettings', 'FinalSuffixoidFile'))
    y_suff_dict_total = dc.create_affixoid_dictionary(config.get('FileSettings', 'FinalSuffixoidFile'), 'Y')
    n_suff_dict_total = dc.create_affixoid_dictionary(config.get('FileSettings', 'FinalSuffixoidFile'), 'N')
    y_suff_dictionary = dc.create_empty_dictionary(config.get('FileSettings', 'FinalSuffixoidFile'))
    n_suff_dictionary = dc.create_empty_dictionary(config.get('FileSettings', 'FinalSuffixoidFile'))
    print('Total:\t', len(suff_inventory_list))

    for m in methods:
        print('=' * 40)
        print(str(m))
        y_suff_dict = y_suff_dictionary.copy()
        n_suff_dict = n_suff_dictionary.copy()
        for k in y_suff_dict.keys():
            c_y = 0
            c_n = 0
            for p in suff_inventory_list:
                if k == p[-3]:
                    frequency = m(p[0])
                    if frequency > 0:
                        if p[-1] == 'Y':
                            c_y += 1
                            y_suff_dict.update({p[-3]: c_y})
                        if p[-1] == 'N':
                            c_n += 1
                            n_suff_dict.update({p[-3]: c_n})
                    else:
                        pass
                else:
                    pass

        print('Y:\t', y_suff_dict)
        print('N:\t', n_suff_dict)
        print()

    st.plot_statistics(y_pref_dict_total, n_pref_dict_total, 'Prefixoid')
    st.plot_statistics(y_suff_dict_total, n_suff_dict_total, 'Suffixoid')
