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
import requests
import sys
import time
import urllib.parse

from pygermanet import load_germanet
from nltk.corpus import stopwords
from nltk.tokenize import load
from nltk.tokenize import RegexpTokenizer
# nltk.download('punkt')
# nltk.download('stopwords')

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

# Sketch Engine
s = requests.Session()
base_url = 'https://api.sketchengine.co.uk/bonito/run.cgi'
corpname = 'sdewac2'
username = 'spahic'
api_key = '3c09af0e68784050a71a5aa8be81d544'
method = '/view'

""" WSD Dictionaries """
PREF_JSON_DICT = 'pref_dictionary.json'
SUFF_JSON_DICT = 'suff_dictionary.json'


class Wsd:
    """ Wsd Class

        Returns: Files with feature vectors

        Example: PREF = Wsd()

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
                >>> self.read_file_to_list(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE)

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
        """ Helper function to read a json file. """
        j = open(json_file, 'r', encoding='utf-8')
        json_data = json.load(j)

        return json_data

    def transform_class_name_to_binary(self, class_name):
        """ This function transforms class labels to binary indicators

            Args:
                class_name (str): Class label (Y|N)

            Returns:
                Binary indicator for class label [0,1]

            Example:
                >>> self.transform_class_name_to_binary('Y')

        """

        if class_name == 'Y':
            return 1

        if class_name == 'N':
            return 0

        else:
            print(Style.BOLD + 'Class Label not known. Exiting program' + Style.END)
            sys.exit()

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

    def is_in_germanet(self, word):
        """ This function parses GermaNet for a word and returns a boolean if the word is found """

        if len(ger.synsets(word)) > 0:
            return True
        else:
            return False

    def search_germanet_synsets(self, word):
        """ Helper function which returns a list of Synsets for a given word """
        return ger.synsets(word)

    def create_synsets_dictionary(self, word, return_lemmas=True, return_hypernyms=True, return_hyponyms=True, return_wiktionary_sense=True):
        """ TODO """

        synsets = ger.synsets(word)
        synset_dict = {}
        for syn in synsets:

            lemmas = []
            hypernyms = []
            hyponyms = []
            wiktionary_sense = []

            if return_lemmas:
                for l in syn.lemmas:
                    lemmas.append(l.orthForm)

            if return_hypernyms:
                for h in syn.hypernyms:
                    for l in h.lemmas:
                        hypernyms.append(l.orthForm)

            if return_hyponyms:
                for h in syn.hyponyms:
                    for l in h.lemmas:
                        hyponyms.append(l.orthForm)

            if return_wiktionary_sense:
                for l in syn.lemmas:
                    for x in l.paraphrases:
                        wiktionary_sense.append(x['wiktionarySense'])

            features = [lemmas, hypernyms, hyponyms, wiktionary_sense]
            synset_dict.update({syn: [item for sublist in features for item in sublist]})

        return synset_dict

    def calculate_similarity_scores(self, word_1, word_2):
        """ This function calculates similarity scores between two synsets """

        word_1_senses = ger.synsets(word_1)
        word_2_senses = ger.synsets(word_2)

        print(word_1_senses)
        print(word_2_senses)

        if len(word_1_senses) > 0 and len(word_2_senses) > 0:
            for p in itertools.product(word_1_senses, word_2_senses):
                print('===')
                print('Hypernym distances for', word_1, len(p[0].hypernym_distances), sorted(list(p[0].hypernym_distances), key=lambda tup: tup[1]))
                print('Hypernym distances for:', word_2, len(p[1].hypernym_distances), sorted(list(p[1].hypernym_distances), key=lambda tup: tup[1]))
                # print(len(p[0].hypernym_paths), p[0].hypernym_paths)
                # print(len(p[1].hypernym_paths), p[1].hypernym_paths)
                print('LCH:', p[0], p[1], '=', p[0].sim_lch(p[1]))
                print('RES:', p[0], p[1], '=', p[0].sim_res(p[1]))
                print('JCN:', p[0], p[1], '=', p[0].dist_jcn(p[1]))
                print('LIN:', p[0], p[1], '=', p[0].sim_lin(p[1]))
                print('---')
                print('Common hypernyms:', len(p[0].common_hypernyms(p[1])), p[0].common_hypernyms(p[1]))
                print('Lowest common hypernyms:', p[0].lowest_common_hypernyms(p[1]))
                print('Nearest common hypernyms:', p[0].nearest_common_hypernyms(p[1]))
                print('Shortest path lenght:', p[0].shortest_path_length(p[1]))
                print('===')

    def lesk(self, ambigous_part_of_word, full_word, remove_stop_words=False, join_sense_and_example=False):
        """ TODO: fix this function """

        score_sense_0 = 0
        score_sense_1 = 0

        try:
            # Sense 0 ambigous_word
            sense_0 = self.definition_dict[ambigous_part_of_word]['0']
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
            sense_1 = self.definition_dict[ambigous_part_of_word]['1']
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
            sense_1_bedeutung_words_distinct = set(sense_1_bedeutung_words_clean)

            sense_1_beispiel_words = word_tok.tokenize(sense_1_beispiel)
            sense_1_beispiel_words_clean = [w for w in sense_1_beispiel_words if w not in stop_words]
            sense_1_beispiel_words_distinct = set(sense_1_beispiel_words_clean)

            # Other Word
            full_word_context = self.get_sentence_for_word(full_word)
            # other_word_bedeutung = [list(s.values()) for s in full_word[1:]]
            # other_word_bedeutung = ' '.join([item for sublist in other_word_bedeutung for item in sublist if item is not None])
            other_word_bedeutung_words = word_tok.tokenize(full_word_context)
            other_word_bedeutung_words_clean = [w for w in other_word_bedeutung_words if w not in stop_words]

            other_word_bedeutung_words_distinct = set(other_word_bedeutung_words_clean)

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

            # print('Overlap in sense_N', len(overlap_sense_0), overlap_sense_0)
            # print('Overlap in sense_Y', len(overlap_sense_1), overlap_sense_1)
            # print('Overlap in Beisp_N', len(overlap_sense_0_beispiel), overlap_sense_0_beispiel)
            # print('Overlap in Beisp_Y', len(overlap_sense_1_beispiel), overlap_sense_1_beispiel)

            score_sense_0 += len(overlap_sense_0)
            score_sense_0 += len(overlap_sense_0_beispiel)
            score_sense_1 += len(overlap_sense_1)
            score_sense_1 += len(overlap_sense_1_beispiel)

        except KeyError:
            pass

        if score_sense_0 > score_sense_1:
            return 0
        if score_sense_1 > score_sense_0:
            return 1
        if score_sense_0 == 0 and score_sense_1 == 0:
            return 1
        else:
            return 0

    def get_sentence_for_word(self, word):
        """ TODO """
        """
            if you want to do fewer than 50 requests, you don’t need to use any waiting,
            if you want to do up to 900 requests, you need to use the interval of 4 seconds per query,
            if you want to do more than 2000 requests, you need to use interval ca 44 seconds.
        """

        attrs = dict(corpname=corpname, q='', pagesize='200', format='json', username=username, api_key=api_key, viewmode='sentence', lpos= '-n', kwicleftctx=20, kwicrightctx=20)
        attrs['q'] = 'q' + '[lemma="'+word+'"]'
        # encoded_attrs = urllib.parse.urlencode(attrs)
        url = base_url + method
        # The requests module can handle building the url parameter stuff
        # We just give it a dictionary (attrs)
        r = s.get(url, params=attrs)
        # print(r)

        # json data stuff
        # the requests module also handles the json output nicely. ;)
        json_obj = r.json()
        # print(json_obj)
        result_count = int(json_obj.get('concsize', '0'))
        # print(word + '\t' + str(json_obj.get('concsize', '0')))
        text = ''
        if result_count > 0:
            response = json.dumps(json_obj["Lines"], sort_keys=True, indent=4, ensure_ascii=False)
            item_dict = json.loads(response)
            sentences_count = len(item_dict)
            if sentences_count > 0:
                counter = 0
                while counter < sentences_count:
                    left = ''
                    kwic = item_dict[counter]['Kwic'][0]['str']
                    right = ''

                    try:
                        left = item_dict[counter]['Left'][0]['str']
                    except IndexError:
                        pass

                    try:
                        right = item_dict[counter]['Right'][0]['str']
                    except IndexError:
                        pass

                    text += left + kwic + right + ' '

                    counter += 1
        return text

class Style:
    """ Helper class for nicer coloring """
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
    pref_inventory_list = PREF_WSD.read_file_to_list(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE)
    f0_pref_wsd_labels = []  # wsd labels
    f1_pref_wsd_list = []  # wsd feature

    """ Loop """
    counter = 0
    for i in pref_inventory_list:
        counter += 1
        if counter == 500:
            break
        elif counter <= 450:
            pass
        # elif counter % 3 == 0:
        #     pass
        else:
            # print('Line:', str(counter) + ' ===============================', i[0], i[-1])
            f0 = PREF_WSD.transform_class_name_to_binary(i[-1])
            f1 = PREF_WSD.lesk(PREF_WSD.split_word_at_pipe(i[1])[0], i[0])
            # print(PREF_WSD.create_synsets_dictionary(PREF_WSD.split_word_at_pipe(i[1])[1]))
            # PREF.calculate_similarity_scores(PREF.split_word_at_pipe(i[1])[0], PREF.split_word_at_pipe(i[1])[1])
            f0_pref_wsd_labels.append(f0)
            f1_pref_wsd_list.append(f1)

    PREF_WSD.write_list_to_file(f0_pref_wsd_labels, DATA_FEATURES_PATH + 'f0_pref_wsd.txt')
    PREF_WSD.write_list_to_file(f1_pref_wsd_list, DATA_FEATURES_PATH + 'f1_pref_wsd.txt')

    """ Tests """
    # print(PREF_WSD.is_in_germanet('Test'))
    # print(PREF_WSD.search_germanet_synsets('Bilderbuch'))
    # print(PREF_WSD.create_synsets_dictionary('Bilderbuch'))
    # print(PREF_WSD.calculate_similarity_scores('Bilderbuch', 'Auflage'))
    # print(PREF_WSD.get_sentence_for_word('Traumstrand'))
    # print(PREF_WSD.get_sentence_for_word('Traumsymbol'))

    # print(PREF_WSD.lesk('Bilderbuch', 'Bilderbuchauftakt'))
    # print(PREF_WSD.lesk('Bombe', 'Bombenprozeß'))
    # print(PREF_WSD.lesk('Bombe', 'Bombenspezialist'))

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

    # for i in inventory:
    #     print(PREF_WSD.create_synsets_dictionary(i))
