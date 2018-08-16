#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
file_reader

Short description:
This module writes various file formats

License: MIT License
Version: 1.0

"""
import io


def write_list_to_file(input_list, output_file):
    """ This function reads a list with affixoids or features and writes lines to a file

        Args:
            input_list (list): List with affixoid instances or features
            output_file (file): Output file

        Returns:
            Output file

        Example:
            >>> write_list_to_file(['Bilderbuch', 'Absturz'], 'doctests/out.txt')
            File written to: doctests/out.txt

    """

    f = open(output_file, 'w', encoding='utf-8')

    for item in input_list:
        if type(item) == list:
            sublist = item
            output_line = '\t'.join([str(x) for x in sublist])
        else:
            output_line = item

        f.write(str(output_line) + '\n')

    print('File written to:', output_file)

    f.close()


def write_dict_to_file(dictionary, output_file):
    """ This function writes a dictionary to a file.

        Args:
            dictionary (dict): Dictionary
            output_file (file): Output file

        Returns:
            Output file

        Example:
            >>> write_dict_to_file({'Bilderbuch': [0, 0]}, 'doctests/dict_out.txt')
            Dictionary written to: doctests/dict_out.txt

    """

    with io.open(output_file, 'w', encoding='utf8') as data:
        data.write(str(dictionary))

    print('Dictionary written to:', output_file)
