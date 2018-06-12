#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
affixoid_statistics

Short description:
This module writes various statistics about the affixoids_inventory.

License: MIT License
Version: 1.0

"""

# Import dependencies
import numpy as np
import matplotlib.pyplot as plt

DATA_FILES_PATH = '../../data/'
DATA_FILES_OUTPUT_PATH = '../../data/statistics/'

affixoids_inventory = []
affixoids_counter_list = []
affixoids_Y_counter_list = []
affixoids_N_counter_list = []

def affixoid_statistics(affixoid_inventory, data_file_path):
    """ This function iterates over txt files and writes various statistics
        about the cuewords, scope, focus and negated targets.

        Args:
            affixoid_inventory (str): Path to input file with affixoid inventory
            data_file_path (str): Path to input file with annotated data

        Returns:
            Written txt file for each file in the input path.

        Example:
            >>> affixoid_statistics(DATA_FILES_PATH + 'prefixoid_inventory.txt',
            DATA_FILES_PATH + 'binary_unique_instance_label_pairs_prefixoids.csv.affixoidal_status.tsv')

    """

    print('Extracting affixoid statistics from:', data_file_path, 'to:', DATA_FILES_OUTPUT_PATH)

    with open(affixoid_inventory, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            affixoids_inventory.append(line)

    print('List of Affixoids: ', affixoids_inventory)
    print('Length of Affixoid candidates: ', len(affixoids_inventory))

    for a in affixoids_inventory:
        line_counter = 0
        Y_counter = 0
        N_counter = 0
        with open(data_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split()
                if line[1].lower().startswith(a.lower()):
                    line_counter += 1
                    if len(line) < 4:
                        print('Missing data for: ', line)
                    if line[3] == 'Y':
                        Y_counter += 1
                    if line[3] == 'N':
                        N_counter += 1
            affixoids_counter_list.append(line_counter)
            affixoids_Y_counter_list.append(Y_counter)
            affixoids_N_counter_list.append(N_counter)

    with open(data_file_path, 'r', encoding='utf-8') as f:
        line_counter = 0

        for line in f:
            line_counter += 1

    print('Data length: ', line_counter)
    print('Data per candidate: ', affixoids_counter_list)
    print('Y candidate: ', affixoids_Y_counter_list)
    print('N candidate: ', affixoids_N_counter_list)
    print(sum(affixoids_counter_list))


def plot_statistics(title):

    N = len(affixoids_inventory)
    y_candidates = affixoids_Y_counter_list

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, y_candidates, width, color='y')

    n_candidates = affixoids_N_counter_list
    rects2 = ax.bar(ind + width, n_candidates, width, color='r')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Counts')
    ax.set_title('Counts per '+title+' candidate')
    ax.set_xticks(ind + width)
    ax.set_xticklabels((affixoids_inventory))

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


if __name__ == "__main__":
    print('Extracting prefixoids:')
    affixoid_statistics(DATA_FILES_PATH + 'prefixoid_inventory.txt',
                        DATA_FILES_PATH + 'binary_unique_instance_label_pairs_prefixoids.csv.affixoidal_status.tsv')
    plot_statistics('prefixoid')

    print('--------------------')

    # empty lists
    affixoids_inventory = []
    affixoids_counter_list = []
    affixoids_Y_counter_list = []
    affixoids_N_counter_list = []

    print('Extracting suffixoids:')
    affixoid_statistics(DATA_FILES_PATH + 'suffixoid_inventory.txt',
                        DATA_FILES_PATH + 'binary_unique_instance_label_pairs_suffixoids.csv.affixoidal_status.tsv')
    plot_statistics('suffixoid')

