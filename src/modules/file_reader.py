#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
file_reader

Short description:
This module reads various file formats

License: MIT License
Version: 1.0

"""
import ast
import json
import sys


def read_file_to_list(affixoid_file):
    """ This function reads a tab-separated file with affixoids and returns a list of lines from file

        Args:
            affixoid_file (file): File with affixoid instances

        Returns:
            List of lines from file

        Example:
            >>> read_file_to_list('doctests/affixoid_file.txt')
            [['Bilderbuchabsturz', 'Bilderbuch|Absturz', 'Bilderbuch', 'Ideal', 'Y']]

    """

    file_as_list = []

    try:
        with open(affixoid_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                if len(word) > 1:
                    file_as_list.append(word)
                else:
                    file_as_list.append(word[0])

        return file_as_list

    except FileNotFoundError:
        print('Error! Can\'t find:', affixoid_file)
        sys.exit('File not found')


def read_dict_from_file(dictionary_file):
    """ This function reads a file with dictionary entries in rb mode and returns dictionary object

        Args:
            dictionary_file (file): Dictionary file

        Returns:<
            Dictionary object

        Example:
            >>> read_dict_from_file('doctests/dictionary.txt')
            {'Bilderbuch': [0, 0]}

    """

    try:
        with open(dictionary_file, "rb") as data:
            dictionary = ast.literal_eval(data.read().decode('utf-8'))

        return dictionary

    except FileNotFoundError:
        print('Error! Can\'t find:', dictionary_file)
        sys.exit('Dictionary not found')


def read_json_from_file(json_file):
    """ This function reads a JSON file and returns a JSON object

        Args:
            json_file (file): JSON formatted file

        Returns:
            JSON object

        Example:
            >>> read_json_from_file('doctests/file.json')
            {'test': 1}

    """

    try:
        j = open(json_file, 'r', encoding='utf-8')
        json_data = json.load(j)

        return json_data

    except FileNotFoundError:
        print('Error! Can\'t find:', json_file)
        sys.exit('JSON file not found')


def read_features_from_files(feature_files_list, path):
    """ This function reads features from files, zips them to a list and returns the list.

        Args:
            feature_files_list (list): List with features
            path (path): Path to the folder with features

        Returns:
            List with zipped features from feature_files_list

        Example:
            >>> read_features_from_files(['f1.txt', 'f2.txt'], path='doctests/')
            [[1, 4853]]

    """

    feature_instances = []
    files = []

    try:
        for file in feature_files_list:
            f = open(path+file, 'r', encoding='utf-8')
            files.append(f)

        zipped_files = zip(*files)

        for line in zipped_files:
            feature_vector = []
            for t in line:
                vec = t.split()
                for v in vec:
                    item = ast.literal_eval(v)
                    feature_vector.append(item)
            feature_instances.append(feature_vector)

        return feature_instances

    except FileNotFoundError:
        print('Error! Can\'t find:', feature_files_list)
        sys.exit('Files not found. Please set correct path to feature lists.')


def read_labels_from_file(file, path):
    """ This function reads features from files, zips them to a list and returns the list.

        Args:
            file (file): File with labels
            path (path): Path to the folder with labels

        Returns:
            List with labels

        Example:
            >>> read_labels_from_file('f0.txt', path='doctests/')
            [1]

    """

    labels = []

    try:

        with open(path+file, 'r', encoding='utf-8') as feat_1:
            for line in feat_1:
                item = ast.literal_eval(line)
                labels.append(item)

        return labels

    except FileNotFoundError:
        print('Error! Can\'t find:', file)
        sys.exit('Files not found. Please set correct path to labels lists.')
