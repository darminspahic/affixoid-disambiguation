#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
Wsd

Short description:
TODO

License: MIT License
Version: 1.0

"""

import itertools
import json
import re
from pygermanet import load_germanet
from nltk.corpus import stopwords
from nltk.tokenize import load
from nltk.tokenize import RegexpTokenizer

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

################
# GermaNet & WordNet
################
ger = load_germanet()

# Sentence tokenizer
sent_tok = load('tokenizers/punkt/german.pickle')

# Filter stopwords
stop_words = set(stopwords.words('german'))

# Word Tokenizer
# word_tok = TreebankWordTokenizer()
word_tok = RegexpTokenizer(r'\w+')

"""WSD Dictionaries"""
PREF_JSON_DICT = 'pref_dictionary.json'
SUFF_JSON_DICT = 'suff_dictionary.json'


class Wsd:
    """ Wsd Class

        Returns: Files with feature vectors

        Example: PREF = FeatureExtractor()

    """

    def __init__(self, string, json_dict):
        print('=' * 40)
        print(Style.BOLD + "Running word sense disambiguation on:" + Style.END, string)
        print('-' * 40)

        try:
            print('Initializing dictionary...')
            self.definition_dict = self.read_json_from_file(json_dict)

        except FileNotFoundError:
            print('Please set correct paths for data.')

    def read_file_to_list(self, affixoid_file):
        """ This function reads a tab-separated file with affixoids and returns a list of lines from file

            Args:
                affixoid_file (file): File with affixoid instances

            Returns:
                List of lines from file

            Example:
                >>> self.read_file_to_list(DATA_FINAL_PATH+FINAL_PREFIXOID_FILE)

        """
        file_as_list = []

        with open(affixoid_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                if len(word) > 1:
                    file_as_list.append(word)
                else:
                    file_as_list.append(word[0])

        return file_as_list

    def write_list_to_file(self, input_list, output_file, item_range=-1, split_second_word=False):
        """ This function reads a list with affixoids or features and writes lines to a file

            Args:
                input_list (list): List with affixoid instances or features
                output_file (file): Output file
                item_range (int): indicator to which index the list returns a line
                split_second_word (bool): Split second word in lists ['Abfalldreck', 'Abfall|Dreck', 'Dreck', 'Schmutz', 'N']

            Returns:
                Output file

            Example:
                >>> self.write_list_to_file(['Bilderbuch', 'Absturz'], 'out.txt')

        """

        f = open(output_file, 'w', encoding='utf-8')

        for item in input_list:
            if split_second_word:
                split_word = self.split_word_at_pipe(item[1])
                output_line = item[0] + '\t' + split_word[0] + '\t' + split_word[1]

            else:
                if type(item) == list:
                    sublist = item
                    output_line = '\t'.join([str(x) for x in sublist])
                else:
                    output_line = item

            f.write(str(output_line) + '\n')

        print(Style.BOLD + 'File written to:' + Style.END, output_file)

        f.close()

    def read_json_from_file(self, json_file):
        j = open(json_file, 'r', encoding='utf-8')
        json_data = json.load(j)

        return json_data

    def split_word_at_pipe(self, word):
        """ This function splits a word separated by a | symbol

            Args:
                word (str): Word with a pipe symbol

            Returns:
                A list of split items

            Examples:
                >>> self.split_word_at_pipe('Bilderbuch|Absturz')

        """

        if '|' in word:
            return word.split('|')

        else:
            return [word, word]

    def search_germanet_synonyms(self, word):

        lemmas = []
        reg_1 = r"\(([^)]+)\)"

        word_synsets = ger.synsets(word)
        match_word = re.findall(reg_1, str(word_synsets))

        for m1 in match_word:
            s1 = ger.synset(m1)
            lemmas.append([s.orthForm for s in s1.lemmas])

        return lemmas

    def search_germanet_synsets(self, word):
        return ger.synsets(word)

    def return_germanet_lemmas(self, word):
        return ger.lemmas(word)

    def search_germanet_similarities(self, word_1, word_2):
        """ TODO """

        # print('Searching for: ', word)

        reg_1 = r"\(([^)]+)\)"
        reg_2 = r"(?<=\()(.*?)(?=\.)"
        word_1_synset = ger.synsets(word_1)
        word_2_synset = ger.synsets(word_2)
        # print(word_synset)
        match_word_1 = re.findall(reg_1, str(word_1_synset))
        match_word_2 = re.findall(reg_1, str(word_2_synset))

        match_word_1_synset_list = []
        match_word_2_synset_list = []

        match_word_1_lemmas_list = []
        match_word_2_lemmas_list = []
        """
        
        b = ger.synset('Heide.n.1')
        a = ger.synset('Graben.n.1')
        
        print(b.lowest_common_hypernyms(a))
        print(b.shortest_path_length(a))
        print(b.sim_lch(a))
        print(b.sim_res(a))
        print(b.sim_lin(a))
        """

        # print(match_word_1)
        # print(match_word_2)

        print(self.search_germanet_definitions(word_1))
        print(self.search_germanet_definitions(word_2))

        for m1 in match_word_1:
            s1 = ger.synset(m1)
            match_word_1_synset_list.append(s1)
            match_word_1_lemmas_list.append(s1.lemmas)

        for m2 in match_word_2:
            s2 = ger.synset(m2)
            match_word_2_synset_list.append(s2)
            match_word_2_lemmas_list.append(s2.lemmas)

        print('Synsets for:', word_1, match_word_1_synset_list)
        print('Synsets for:', word_2, match_word_2_synset_list)

        print('Lemmas for:', word_1, match_word_1_lemmas_list)
        print('Lemmas for:', word_2, match_word_2_lemmas_list)

        if len(match_word_1_synset_list) > 0 and len(match_word_2_synset_list) > 0:
            for p in itertools.product(match_word_1_synset_list, match_word_2_synset_list):
                print('===')
                print(len(p[0].hypernym_distances), sorted(list(p[0].hypernym_distances), key=lambda tup: tup[1]))
                print(len(p[1].hypernym_distances), sorted(list(p[1].hypernym_distances), key=lambda tup: tup[1]))
                # print(len(p[0].hypernym_paths), p[0].hypernym_paths)
                # print(len(p[1].hypernym_paths), p[1].hypernym_paths)
                print('LCH:', p[0], p[1], '=', p[0].sim_lch(p[1]))
                print('RES:', p[0], p[1], '=', p[0].sim_res(p[1]))
                print('JCN:', p[0], p[1], '=', p[0].dist_jcn(p[1]))
                print('LIN:', p[0], p[1], '=', p[0].sim_lin(p[1]))
                print('---')
                print('Common hypernyms:', p[0].common_hypernyms(p[1]), len(p[0].common_hypernyms(p[1])))
                print('Lowest common hypernyms:', p[0].lowest_common_hypernyms(p[1]))
                print('Nearest common hypernyms:', p[0].nearest_common_hypernyms(p[1]))
                print('Shortest path lenght:', p[0].shortest_path_length(p[1]))
                print('===')

        else:
            print('Synset not found')

    def search_germanet_definitions(self, word):
        pass
        """ TODO """

        # print('Searching for: ', word)

        definitions = [word]

        if self.is_in_germanet(word):
            for synset in GN_ROOT:
                orthforms = synset.findall('.//orthForm')
                for item in orthforms:
                    if word == item.text or word == item.text.lower() or word == item.text.lower().capitalize():
                        parent = item.getparent()
                        parent_id = item.getparent().get('id')
                        # print(parent)
                        # print('ID:', parent_id)
                        ancestor = parent.getparent()
                        ancestor_id = parent.getparent().get('id')
                        # print(ancestor)
                        # print('ID:', ancestor_id)
                        # TODO get sense of synset id 1, 2 etc.
                        # print('Synset by ID:', ger.get_lemma_by_id(ancestor_id))
                        parent_sense = item.getparent().get('sense')
                        parent_description = PARAROOT.find('.//wiktionaryParaphrase[@lexUnitId = "'+parent_id+'"]')
                        try:
                            sense_dict = {parent_sense: parent_description.get('wiktionarySense')}
                        except AttributeError:
                            sense_dict = {parent_sense: None}
                        definitions.append(sense_dict)

        if len(definitions) > 0:
            return definitions
        else:
            return 0

    def lesk(self, ambigous_word, other_word, remove_stop_words=False, join_sense_and_example=False):
        try:
            # Sense 0 ambigous_word
            sense_0 = self.definition_dict[ambigous_word]['0']
            sense_0_germanet = sense_0['GermaNet']['Bedeutung']
            sense_0_wiktionary = sense_0['Wiktionary']['Bedeutung']
            sense_0_duden = sense_0['Duden']['Bedeutung']

            sense_0_germanet_example = sense_0['GermaNet']['Beispiel']
            sense_0_wiktionary_example = sense_0['Wiktionary']['Beispiel']
            sense_0_duden_example = sense_0['Duden']['Beispiel']

            sense_0_bedeutung = sense_0_germanet + ' ' + sense_0_wiktionary + ' ' + sense_0_duden
            sense_0_beispiel = sense_0_germanet_example + ' ' + sense_0_wiktionary_example + ' ' + sense_0_duden_example

            sense_0_bedeutung_words = word_tok.tokenize(sense_0_bedeutung)
            sense_0_bedeutung_words_clean = [w for w in sense_0_bedeutung_words if w not in stop_words]
            sense_0_bedeutung_words_distinct = set(sense_0_bedeutung_words_clean)

            sense_0_beispiel_words = word_tok.tokenize(sense_0_beispiel)
            sense_0_beispiel_words_clean = [w for w in sense_0_beispiel_words if w not in stop_words]
            sense_0_beispiel_words_distinct = set(sense_0_beispiel_words_clean)

            # Sense 1 ambigous_word
            sense_1 = self.definition_dict[ambigous_word]['1']
            sense_1_germanet = sense_1['GermaNet']['Bedeutung']
            sense_1_wiktionary = sense_1['Wiktionary']['Bedeutung']
            sense_1_duden = sense_1['Duden']['Bedeutung']

            sense_1_germanet_example = sense_1['GermaNet']['Beispiel']
            sense_1_wiktionary_example = sense_1['Wiktionary']['Beispiel']
            sense_1_duden_example = sense_1['Duden']['Beispiel']

            sense_1_bedeutung = sense_1_germanet + ' ' + sense_1_wiktionary + ' ' + sense_1_duden
            sense_1_beispiel = sense_1_germanet_example + ' ' + sense_1_wiktionary_example + ' ' + sense_1_duden_example

            sense_1_bedeutung_words = word_tok.tokenize(sense_1_bedeutung)
            sense_1_bedeutung_words_clean = [w for w in sense_1_bedeutung_words if w not in stop_words]
            sense_1_bedeutung_words_distinct = set(sense_1_bedeutung_words)

            sense_1_beispiel_words = word_tok.tokenize(sense_1_beispiel)
            sense_1_beispiel_words_clean = [w for w in sense_1_beispiel_words if w not in stop_words]
            sense_1_beispiel_words_distinct = set(sense_1_beispiel_words)

            # Other Word
            other_word = self.search_germanet_definitions(other_word)
            other_word_bedeutung = [list(s.values()) for s in other_word[1:]]
            other_word_bedeutung = ' '.join([item for sublist in other_word_bedeutung for item in sublist if item is not None])
            other_word_bedeutung_words = word_tok.tokenize(other_word_bedeutung)
            other_word_bedeutung_words_clean = [w for w in other_word_bedeutung_words if w not in stop_words]

            other_word_bedeutung_words_distinct = set(other_word_bedeutung_words)

            # print(sense_0_bedeutung_words_distinct)
            # print(sense_0_beispiel_words_distinct)
            # print()
            # print(sense_1_bedeutung_words_distinct)
            # print(sense_1_beispiel_words_distinct)
            # print()
            # print(other_word_bedeutung_words_distinct)

            overlap_sense_0 = sense_0_bedeutung_words_distinct.intersection(other_word_bedeutung_words_distinct)
            overlap_sense_1 = sense_1_bedeutung_words_distinct.intersection(other_word_bedeutung_words_distinct)

            overlap_sense_0_beispiel = sense_0_beispiel_words_distinct.intersection(other_word_bedeutung_words_distinct)
            overlap_sense_1_beispiel = sense_1_beispiel_words_distinct.intersection(other_word_bedeutung_words_distinct)

            print('Overlap in sense_N', len(overlap_sense_0), overlap_sense_0)
            print('Overlap in sense_Y', len(overlap_sense_1), overlap_sense_1)
            print('Overlap in Beisp_N', len(overlap_sense_0_beispiel), overlap_sense_0_beispiel)
            print('Overlap in Beisp_Y', len(overlap_sense_1_beispiel), overlap_sense_1_beispiel)

        except KeyError:
            pass


class Style:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    UNDERLINE = '\033[4m'
    BOLD = '\033[1m'
    END = '\033[0m'


if __name__ == "__main__":
    """
        PREFIXOIDS WSD
    """
    PREF_WSD = Wsd('Prefixoids', DATA_PATH + 'wsd/' + PREF_JSON_DICT)

    # print(PREF.lesk('Bilderbuch', 'Absturz', pref_json_dict))
    # counter = 0
    # for i in pref_inventory_list:
    #     counter += 1
    #     if counter == 310:
    #         break
    #     elif counter <= 210:
    #         pass
    #     elif counter % 3 == 0:
    #         pass
    #     else:
    #     print('Line:', str(counter) + ' ===============================', i[0], i[-1])
    #     # PREF.search_germanet_similarities(PREF.split_word_at_pipe(i[1])[0], PREF.split_word_at_pipe(i[1])[1])
    #     PREF_WSD.lesk(PREF_WSD.split_word_at_pipe(i[1])[0], PREF_WSD.split_word_at_pipe(i[1])[1], pref_json_dict)

    # print(PREF_WSD.search_germanet_similarities('Bilderbuch', 'Bilderbuch', 'N'))
    # print(PREF_WSD.search_germanet_similarities('Pferd', 'Hengst', 'N'))
    # print(PREF_WSD.search_germanet_definitions('Pferd'))
    # print(PREF_WSD.search_germanet_definitions('Hengst'))
    # print(PREF_WSD.search_germanet_similarities('Kohle', 'Tagebuch'))
    # print(PREF_WSD.search_germanet_similarities('Hund', 'Katze'))
    # print(PREF_WSD.search_germanet_definitions('Wunsch'))
    # print(PREF_WSD.search_germanet_definitions('Kohle'))
    print(PREF_WSD.search_germanet_synonyms('Geld'))
    print()
    print(PREF_WSD.return_germanet_lemmas('Geld'))
    print()
    print(PREF_WSD.search_germanet_synsets('Geld'))

    inventory = ['Bilderbuch',
                 'Blitz',
                 'Bombe',
                 'Glanz',
                 'Heide',
                 'Jahrhundert',
                 'Qualität',
                 'Schwein',
                 'Spitze',
                 'Traum',
                 'Apostel',
                 'Bolzen',
                 'Dreck',
                 'Gott',
                 'Guru',
                 'Hengst',
                 'Ikone',
                 'König',
                 'Papst',
                 'Schwein']

    """
    <synset id="s38531" category="nomen" class="Mensch">
    
        <lexUnit id="l118841" sense="1" source="core" namedEntity="no" artificial="no" styleMarking="no">
            <orthForm>Gottlose</orthForm>
        </lexUnit>
        
        <lexUnit id="l118842" sense="1" source="core" namedEntity="no" artificial="no" styleMarking="no">
            <orthForm>Gottloser</orthForm>
        </lexUnit>
        
        <lexUnit id="l56466" sense="2" source="core" namedEntity="no" artificial="no" styleMarking="no">
            <orthForm>Heide</orthForm>
        </lexUnit>
        
        <lexUnit id="l56467" sense="1" source="core" namedEntity="no" artificial="no" styleMarking="no">
            <orthForm>Heidin</orthForm>
        </lexUnit>
    </synset>
    """

    # for i in inventory:
    #     print(PREF.search_germanet_definitions(i))
