#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
feature_extractor

Short description:
This module writes various statistics about the affixoids_inventory.

License: MIT License
Version: 1.0

"""

DATA_FILES_PATH = '../../data/'
DATA_FILES_PATH_FINAL = '../../data/final/'
DATA_FILES_OUTPUT_PATH = '../../data/statistics/'


def feature_extractor(affixoid_inventory):
    """ This function

        Args:


        Returns:


        Example:
            >>> feature_extractor(DATA_FILES_PATH + 'binary_unique_instance_label_pairs_prefixoids.csv.affixoidal_status.tsv')

    """

    # Empty lists for collecting data
    # feature vector ['Wort', 'Affixoid|Stammwort', 'Affixoid', 'Bedeutung', 'Klasse']
    # Data example: Bilderbuchabsturz	Bilderbuch|Absturz	Bilderbuch	Ideal	Y
    feature_inventory = []

    with open(affixoid_inventory, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split()
            f_0 = word[0]  # keep full word for easier controlling
            f_1 = transform_to_binary(word[-1])  # feature 1: binary indicator if affixoid
            f_1_1 = word[1]
            f_2 = word[2]
            f_3 = word[3]
            f_4 = word[4]  # keep class name for easier controlling
            feature_inventory.append([f_0, f_1, f_1_1, f_2, f_3, f_4])

    return feature_inventory


def compare_files(affixoid_inventory_1, affixoid_inventory_2, compare=True):

    if compare:
        file_1 = []
        file_2 = []
        with open(affixoid_inventory_1, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                f_0 = word[0]  # keep full word for easier controlling
                file_1.append([f_0])

        with open(affixoid_inventory_2, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                f_0 = word[0]  # keep full word for easier controlling
                file_2.append([f_0])

        counter = 0
        for i in file_1:
            if file_1[counter] == file_2[counter]:
                pass
            else:
                print(file_1[counter], file_2[counter])
            counter+=1

        print(len(file_1))
        print(len(file_2))

    else:
        file_1 = []
        file_2 = []
        file_3 = []
        # aList.insert( 3, 2009)
        with open(affixoid_inventory_1, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                file_1.append(word)

        with open(affixoid_inventory_2, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                file_2.append(word)

        counter = 0
        for i in file_1:
            if file_1[counter][0] == file_2[counter][0]:
                # print(file_1[counter][0], file_2[counter][0])
                print(file_1[counter], file_2[counter])
                new_line = list(file_2[counter] + file_1[counter][1:])
                file_3.append(new_line)

            else:
                exit()
            counter += 1

        f = open('binary_unique_instance_suffixoid_segmentations.txt', 'w', encoding='utf-8')

        for item in file_3:
            # ['Abfalldreck', 'Abfall|Dreck', 'Dreck', 'Schmutz', 'N']
            output_line = item[0]+'\t'+item[1]+'\t'+item[2]+'\t'+item[3]+'\t'+item[4]
            f.write(output_line + "\n")

        f.close()

        print(file_3)
        print(len(file_3))
        # print(file_2)


def transform_to_binary(class_name):
    if class_name == 'Y':
        return 1
    if class_name == 'N':
        return 0
    else:
        print('Class Label not known')


if __name__ == "__main__":
    # prefixoids = feature_extractor(DATA_FILES_PATH_FINAL + 'binary_unique_instance_prefixoid_segmentations.txt')
    # print(prefixoids)
    compare_files(DATA_FILES_PATH + 'binary_unique_instance_label_pairs_prefixoids.csv.affixoidal_status.tsv', DATA_FILES_PATH_FINAL + 'binary_unique_instance_prefixoid_segmentations.txt', True)
    compare_files(DATA_FILES_PATH + 'binary_unique_instance_label_pairs_prefixoids.csv.affixoidal_status.tsv', DATA_FILES_PATH + 'prefixoid_segmentations_smor.txt.manual_splits.tsv', True)
    compare_files(DATA_FILES_PATH + 'binary_unique_instance_label_pairs_suffixoids.csv.affixoidal_status.tsv', DATA_FILES_PATH_FINAL + 'binary_unique_instance_suffixoid_segmentations.txt', True)
    compare_files(DATA_FILES_PATH + 'binary_unique_instance_label_pairs_suffixoids.csv.affixoidal_status.tsv', DATA_FILES_PATH + 'suffixoid_ segmentations_smor.txt.manual_splits.tsv', True)
    # compare_files(DATA_FILES_PATH + 'binary_unique_instance_label_pairs_prefixoids.csv.affixoidal_status.tsv', DATA_FILES_PATH + 'prefixoid_segmentations_smor.txt.manual_splits.tsv', False)
    # compare_files(DATA_FILES_PATH + 'binary_unique_instance_label_pairs_suffixoids.csv.affixoidal_status.tsv', DATA_FILES_PATH + 'suffixoid_segmentations_smor.txt.manual_splits.tsv', False)

