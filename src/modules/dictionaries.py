#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
dictionaries

Short description:
This module manipulates various types of dictionaries

License: MIT License
Version: 1.0

"""

import ast
import bz2
import sys

from modules import file_writer as fw


def create_affixoid_dictionary(affixoid_file, class_name):
    """ This function creates a dictionary with class instances of affixoids

        Args:
            affixoid_file (file): File with affixoid instances
            class_name (str): Class label (Y|N)

        Returns:
            Dictionary with class instances

        Example:
            >>> create_affixoid_dictionary('doctests/affixoid_file.txt', 'Y')
            {'Bilderbuch': 1, 'Blitz': 0}

    """

    dictionary = {}
    counter = 0

    try:
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

    except FileNotFoundError:
        print('Error. File not found:', affixoid_file)
        sys.exit('Files not found. Please set correct path to the affixoid file.')


def create_empty_dictionary(affixoid_file):
    """ This function creates an empty dictionary with affixoids

        Args:
            affixoid_file (file): File with affixoid instances

        Returns:
            Dictionary with class instances

        Example:
            >>> create_empty_dictionary('doctests/affixoid_file.txt')
            {'Bilderbuch': [], 'Blitz': []}

    """

    dictionary = {}

    try:
        with open(affixoid_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                dict_key = word[-3]
                dictionary.update({dict_key: []})

        return dictionary

    except FileNotFoundError:
        print('Error. File not found:', affixoid_file)
        sys.exit('Files not found. Please set correct path to the affixoid file.')


def extract_frequency(word, dictionary, return_as_binary_vector=False):
    """ This function extracts frequencies for a given word from a dictionary of frequencies

        Args:
            word (str): Word
            dictionary (dict): Dictionary with frequencies
            return_as_binary_vector (bool): Returns full vector with binary indicator where the word is found

        Returns:
            A frequency for a given word from a dictionary

        Examples:
            >>> extract_frequency('Bilderbuch', {'Bilderbuch': 1})
            1
            >>> extract_frequency('Bilderbuch', {'Bilderbuch': 1, 'Blitz': 5}, return_as_binary_vector=True)
            [1, 0]

    """

    if return_as_binary_vector:
        dictionary_copy = dictionary.fromkeys(dictionary, 0)
        dictionary_copy.update({word: 1})
        return list(dictionary_copy.values())

    if word in dictionary.keys():
        value = dictionary[word]
        return int(value)

    else:
        return 0


def create_frequency_dictionary(frequency_file):
    """ This function creates a dictionary with frequency instances of affixoids

        Args:
            frequency_file (file): File with affixoid frequencies

        Returns:
            Dictionary with frequency instances of affixoids

        Example:
            >>> create_frequency_dictionary('doctests/frequencies.txt')
            {'Bilderbuchabsturz': '1'}

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


def create_vector_dictionary(vector_file, multiword=False):
    """ This function creates a dictionary with vector values from affixoids

        Args:
            vector_file (file): File with vector values from FastText
            multiword (bool): Set to True if the word in vector file has multiple parts

        Returns:
            Dictionary with vector values as list

        Example:
            >>> create_vector_dictionary('doctests/vectors.txt')
            {'Bilderbuchabsturz': [-0.25007, -0.16484, -0.34915, 0.44351, 0.17918, 0.17356, 0.32336, 0.19306, 0.40586, 0.58886, -0.55027, 0.15365, -0.28948, -0.096226, 0.91019, 0.24468, -0.20271, 0.5475, 0.36233, 0.20612, -0.17727, 0.054958, 0.16082, -0.1237, -0.057176, 0.18833, 0.11853, 0.19447, -0.13197, -0.18862, -0.17965, -0.13153, 0.27431, -0.68191, -0.35592, -0.13321, 0.16669, -0.42519, 0.11905, 0.15686, 0.26408, -0.35616, -0.26065, -0.0021858, 0.34352, -0.39887, 0.59549, -0.35665, -0.60043, -0.16136, -0.19603, -0.57132, 0.11918, -0.22356, 0.1499, -0.22458, -0.081269, 0.0058904, 0.16639, 0.36866, -0.3294, -0.21821, 0.87304, -0.042374, -0.42687, -0.41224, -0.73979, 0.37495, 0.34696, 0.6927, -0.24696, 0.23713, 0.0004817, -0.67652, 0.36679, 0.52095, -0.059838, 0.3779, -0.15106, -0.31892, -0.084559, -0.067978, 0.45779, 0.45037, -0.19661, -0.14229, 0.097991, 0.26219, 0.41556, 0.43363, 0.060991, 0.15759, 0.055367, -0.10719, -0.38255, -0.3, -0.032207, -0.50483, 0.18746, -0.6391]}

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


def create_dictionary_from_word(word):
    """ This function creates a dictionary from splitwords

        Args:
            splitword (str): Word

        Returns:
            Dictionary with word as key and 0 as value

        Example:
            >>> create_dictionary_from_word('Bilderbuch')
            {'Bilderbuch': 0}

    """

    dictionary = {}
    dict_key = word
    dictionary.update({dict_key: 0})

    return dictionary


def create_polarity_dict(polarity_file):
    """ Helper function to create a polarity dictionary, where key = word and value = [vector of values]

        Args:
            polarity_file (file): File with polarity values from SentiMerge

        Returns:
            Dictionary with vector values as list

        Example:
            >>> create_polarity_dict('doctests/polarity.txt')
            {'Aal': '-0.017532794118768923'}

    """

    dictionary = {}

    with open(polarity_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split()
            if word[1] == 'N' or word[1] == 'NE':
                dict_key = word[0].capitalize()
            else:
                dict_key = word[0]
            dict_value = word[2]  # get sentiment value only
            dictionary.update({dict_key: dict_value})

    return dictionary


def extract_dictionary_values(word, polarity_dict):
    """ Helper function to extract polarity for a word from dictionary

        Args:
            word (str):
            polarity_dict (dictionary): Dictionary with values from SentiMerge

        Returns:
            Values as list

        Example:
            >>> extract_dictionary_values('Aal', {'Aal': '-0.017532794118768923'})
            -0.017532794118768923

    """

    if word in polarity_dict.keys():
        value = polarity_dict[word]
        try:
            v = ast.literal_eval(value)
            return v
        except ValueError:
            return value

    else:
        return 0


def extract_pmi_values(pmi_scores, splitwords_dictionary, output_file):
    """ Helper function to extract PMI values for a word from a bz2 file and write values to a file """

    print('Extracting PMI scores...')

    with bz2.BZ2File(pmi_scores, 'r') as pmi_file:
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

    fw.write_dict_to_file(splitwords_dictionary, output_file)

    return splitwords_dictionary
