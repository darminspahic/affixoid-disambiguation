#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
helper_functions

Short description:
This module writes various file formats

License: MIT License
Version: 1.0

"""
import sys


def transform_class_name_to_binary(class_name):
    """ This function transforms class labels to binary indicators

        Args:
            class_name (str): Class label (Y|N)

        Returns:
            Binary indicator for class label [0,1]

        Example:
            >>> transform_class_name_to_binary('Y')
            1
            >>> transform_class_name_to_binary('N')
            0

    """

    if class_name == 'Y':
        return 1

    if class_name == 'N':
        return 0

    else:
        sys.exit('Class Label not known. Exiting program')


def split_word_at_pipe(word):
    """ This function splits a word separated by a | symbol

        Args:
            word (str): Word with a pipe symbol

        Returns:
            A list of split items

        Examples:
            >>> split_word_at_pipe('Bilderbuch|Absturz')
            ['Bilderbuch', 'Absturz']
            >>> split_word_at_pipe('Bilderbuch')
            ['Bilderbuch', 'Bilderbuch']

    """

    if '|' in word:
        return word.split('|')

    else:
        return [word, word]
