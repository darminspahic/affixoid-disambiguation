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
DATA_FINAL_PATH = '../data/final/'
DATA_WSD_PATH = '../data/wsd/'
DATA_WSD_SENTENCES_PATH = '../data/wsd/sentences/'

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
german_stopwords = stopwords.words('german')
german_stopwords.extend(('dass', 'bzw'))
stop_words = set(german_stopwords)

# Word Tokenizer
# word_tok = TreebankWordTokenizer()
word_tok = RegexpTokenizer(r'\w+')

# Sketch Engine
s = requests.Session()
base_url = 'https://api.sketchengine.co.uk/bonito/run.cgi'
corpname = 'sdewac2'
username = 'spahic'
api_key = '159b841f61a64092bc630d20b0f56c93'
# username = 'api_testing'
# api_key = 'YNSC0B9OXN57XB48T9HWUFFLPY4TZ6OE'
method = '/view'
"""
https://www.sketchengine.eu/documentation/api-documentation/
https://www.sketchengine.eu/documentation/methods-documentation/
login: api_testing
api_key: YNSC0B9OXN57XB48T9HWUFFLPY4TZ6OE
"""

""" WSD Dictionaries """
PREF_JSON_DICT = 'pref_dictionary.json'
SUFF_JSON_DICT = 'suff_dictionary.json'

# TODO: add global settings
settings = {
    "return_lemmas": True,
    "return_hypernyms": True,
    "return_hyponyms": True,
    "return_wiktionary_sense": False,
    "remove_stopwords": True,
    "join_sense_and_example": True,
    "use_synonyms": True,
    "lemmatize": False,
    "open_locally": True,
    "write_to_file": False,
    "return_keyword": True,
    "return_single_sentence": False,
}


class Wsd:
    """ Wsd Class

        Returns: Files with feature vectors

        Example: CLSF = Wsd()

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

    def create_affixoid_dictionary(self, affixoid_file, class_name):
        """ This function creates a dictionary with class instances of affixoids

            Args:
                affixoid_file (file): File with affixoid instances
                class_name (str): Class label (Y|N)

            Returns:
                Dictionary with class instances

            Example:
                >>> self.create_affixoid_dictionary(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE, 'Y')

        """
        dictionary = {}
        counter = 0

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

    def create_empty_dictionary(self, affixoid_file):
        """ This function creates a dictionary with class instances of affixoids

            Args:
                affixoid_file (file): File with affixoid instances
                class_name (str): Class label (Y|N)

            Returns:
                Dictionary with class instances

            Example:
                >>> self.create_affixoid_dictionary(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE, 'Y')

        """
        dictionary = {}

        with open(affixoid_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()
                dict_key = word[-3]
                dictionary.update({dict_key: []})

        return dictionary

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

    def write_list_to_file(self, input_list, output_file, split_second_word=False):
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

    def create_synsets_dictionary(self, word,
                                  return_lemmas=settings["return_lemmas"],
                                  return_hypernyms=settings["return_hypernyms"],
                                  return_hyponyms=settings["return_hyponyms"],
                                  return_wiktionary_sense=settings["return_wiktionary_sense"]):
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
                    lemmas.append(l.orthForm.split())

            if return_hypernyms:
                for h in syn.hypernyms:
                    for l in h.lemmas:
                        hypernyms.append(l.orthForm.split())

            if return_hyponyms:
                for h in syn.hyponyms:
                    for l in h.lemmas:
                        hyponyms.append(l.orthForm.split())

            if return_wiktionary_sense:
                for l in syn.lemmas:
                    for x in l.paraphrases:
                        wiktionary_sense.append(x['wiktionarySense'])

            lemmas_flat = [item for sublist in lemmas for item in sublist]
            hypernyms_flat = [item for sublist in hypernyms for item in sublist]
            hyponyms_flat = [item for sublist in hyponyms for item in sublist]
            features = [lemmas_flat, hypernyms_flat, hyponyms_flat, wiktionary_sense]

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

    def lesk(self, ambigous_part_of_word, full_word, n_dict, y_dict,
             remove_stopwords=settings["remove_stopwords"],
             join_sense_and_example=settings["join_sense_and_example"],
             use_synonyms=settings["use_synonyms"],
             lemmatize=settings["lemmatize"]):

        """ TODO: optimize this function """

        score_sense_0 = 0
        score_sense_1 = 0

        try:
            # Sense 0 ambigous_word
            sense_0 = self.definition_dict[ambigous_part_of_word]['0']
            sense_0_germanet = sense_0['GermaNet']['Bedeutung']
            sense_0_wiktionary = sense_0['Wiktionary']['Bedeutung']
            sense_0_duden = sense_0['Duden']['Bedeutung']
            sense_0_crowdsourcing = sense_0['Crowdsourcing']['Bedeutung']

            sense_0_germanet_example = sense_0['GermaNet']['Beispiel']
            sense_0_wiktionary_example = sense_0['Wiktionary']['Beispiel']
            sense_0_duden_example = sense_0['Duden']['Beispiel']

            sense_0_bedeutung = sense_0_germanet + ' ' + sense_0_wiktionary + ' ' + sense_0_duden + ' ' + sense_0_crowdsourcing
            sense_0_beispiel = sense_0_germanet_example + ' ' + sense_0_wiktionary_example + ' ' + sense_0_duden_example

            if remove_stopwords:
                sense_0_bedeutung_words_clean = [w for w in word_tok.tokenize(sense_0_bedeutung) if w.lower() not in stop_words and len(w) > 1]
                sense_0_beispiel_words_clean = [w for w in word_tok.tokenize(sense_0_beispiel) if w.lower() not in stop_words and len(w) > 1]
            else:
                sense_0_bedeutung_words_clean = [w for w in word_tok.tokenize(sense_0_bedeutung)]
                sense_0_beispiel_words_clean = [w for w in word_tok.tokenize(sense_0_beispiel)]

            # Sense 1 ambigous_word
            sense_1 = self.definition_dict[ambigous_part_of_word]['1']
            sense_1_germanet = sense_1['GermaNet']['Bedeutung']
            sense_1_wiktionary = sense_1['Wiktionary']['Bedeutung']
            sense_1_duden = sense_1['Duden']['Bedeutung']
            sense_1_crowdsourcing = sense_1['Crowdsourcing']['Bedeutung']

            sense_1_germanet_example = sense_1['GermaNet']['Beispiel']
            sense_1_wiktionary_example = sense_1['Wiktionary']['Beispiel']
            sense_1_duden_example = sense_1['Duden']['Beispiel']

            sense_1_bedeutung = sense_1_germanet + ' ' + sense_1_wiktionary + ' ' + sense_1_duden + ' ' + sense_1_crowdsourcing
            sense_1_beispiel = sense_1_germanet_example + ' ' + sense_1_wiktionary_example + ' ' + sense_1_duden_example

            if remove_stopwords:
                sense_1_bedeutung_words_clean = [w for w in word_tok.tokenize(sense_1_bedeutung) if w.lower() not in stop_words and len(w) > 1]
                sense_1_beispiel_words_clean = [w for w in word_tok.tokenize(sense_1_beispiel) if w.lower() not in stop_words and len(w) > 1]

            else:
                sense_1_bedeutung_words_clean = [w for w in word_tok.tokenize(sense_1_bedeutung)]
                sense_1_beispiel_words_clean = [w for w in word_tok.tokenize(sense_1_beispiel)]

            # synonyms
            id_0 = self.definition_dict[ambigous_part_of_word]['0']['ID'].split('; ')
            id_1 = self.definition_dict[ambigous_part_of_word]['1']['ID'].split('; ')

            id_0_lists = []
            for i in id_0:
                id_0_lists.append([value for sublist in self.create_synsets_dictionary(i).values() for value in sublist])
            id_0_synonyms = [value for sublist in id_0_lists for value in sublist]

            id_1_lists = []
            for i in id_1:
                id_1_lists.append([value for sublist in self.create_synsets_dictionary(i).values() for value in sublist])
            id_1_synonyms = [value for sublist in id_1_lists for value in sublist]

            # Other Word
            if self.get_sentence_for_word(full_word):
                full_word_context = self.get_sentence_for_word(full_word)
                print('"', full_word_context, '"')
                other_word_bedeutung_words_clean = [w for w in word_tok.tokenize(full_word_context) if w.lower() not in stop_words and len(w) > 1]
            else:
                return -1

            if lemmatize:
                sense_0_bedeutung_words_lemmatized = [ger.lemmatise(w) for w in sense_0_bedeutung_words_clean]
                sense_0_beispiel_words_lemmatized = [ger.lemmatise(w) for w in sense_0_beispiel_words_clean]

                sense_1_bedeutung_words_lemmatized = [ger.lemmatise(w) for w in sense_1_bedeutung_words_clean]
                sense_1_beispiel_words_lemmatized = [ger.lemmatise(w) for w in sense_1_beispiel_words_clean]

                other_word_bedeutung_words_lemmatized = [ger.lemmatise(w) for w in other_word_bedeutung_words_clean]

                sense_0_bedeutung_words_clean = [value for sublist in sense_0_bedeutung_words_lemmatized for value in sublist]
                sense_0_beispiel_words_clean = [value for sublist in sense_0_beispiel_words_lemmatized for value in sublist]

                sense_1_bedeutung_words_clean = [value for sublist in sense_1_bedeutung_words_lemmatized for value in sublist]
                sense_1_beispiel_words_clean = [value for sublist in sense_1_beispiel_words_lemmatized for value in sublist]

                other_word_bedeutung_words_clean = [value for sublist in other_word_bedeutung_words_lemmatized for value in sublist]


            # Calculate overlaps
            overlap_sense_0 = set(other_word_bedeutung_words_clean).intersection(sense_0_bedeutung_words_clean)
            overlap_sense_1 = set(other_word_bedeutung_words_clean).intersection(sense_1_bedeutung_words_clean)

            overlap_sense_0_beispiel = set(other_word_bedeutung_words_clean).intersection(sense_0_beispiel_words_clean)
            overlap_sense_1_beispiel = set(other_word_bedeutung_words_clean).intersection(sense_1_beispiel_words_clean)

            overlap_synonyms_0 = set(other_word_bedeutung_words_clean).intersection(id_0_synonyms)
            overlap_synonyms_1 = set(other_word_bedeutung_words_clean).intersection(id_1_synonyms)

            if join_sense_and_example:
                score_sense_0 += len(overlap_sense_0)
                score_sense_0 += len(overlap_sense_0_beispiel)
                score_sense_1 += len(overlap_sense_1)
                score_sense_1 += len(overlap_sense_1_beispiel)
            else:
                score_sense_0 += len(overlap_sense_0)
                score_sense_1 += len(overlap_sense_1)

            if use_synonyms:
                score_sense_0 += len(overlap_synonyms_0)
                score_sense_1 += len(overlap_synonyms_1)

            print('----')
            print(Style.BOLD + 'Sense_0 Bedeutung:' + Style.END, sense_0_bedeutung)
            print(Style.BOLD + 'Sense_0 Beispiel:' + Style.END, sense_0_beispiel)
            print(Style.BOLD + 'Sense_1 Bedeutung:' + Style.END, sense_1_bedeutung)
            print(Style.BOLD + 'Sense_1 Beispiel:' + Style.END, sense_1_beispiel)
            print(Style.BOLD + 'Example sentence words:' + Style.END, set(other_word_bedeutung_words_clean))
            print('----')
            print(Style.BOLD + 'Synonyms_0:' + Style.END, id_0_synonyms)
            print(Style.BOLD + 'Synonyms_1:' + Style.END, id_1_synonyms)
            print()
            print('Overlap in sense_0:', len(overlap_sense_0), overlap_sense_0)
            print('Overlap in sense_1:', len(overlap_sense_1), overlap_sense_1)
            print('Overlap in Beisp_0:', len(overlap_sense_0_beispiel), overlap_sense_0_beispiel)
            print('Overlap in Beisp_1:', len(overlap_sense_1_beispiel), overlap_sense_1_beispiel)
            print('Overlap in Synon_0:', len(overlap_synonyms_0), overlap_synonyms_0)
            print('Overlap in Synon_1:', len(overlap_synonyms_1), overlap_synonyms_1)
            print()

        except KeyError:
            return -1

        if score_sense_0 > score_sense_1:
            print(Style.BOLD + 'Assigning class: 0' + Style.END)
            return 0
        if score_sense_1 > score_sense_0:
            print(Style.BOLD + 'Assigning class: 1' + Style.END)
            return 1
        if score_sense_0 == 0 and score_sense_1 == 0:
            print(Style.BOLD + 'Assigning class: 1' + Style.END)
            return 1
        else:
            assigned_class = self.return_most_frequent_sense(ambigous_part_of_word, n_dict, y_dict)
            print(Style.BOLD + 'Assigning mfs:' + Style.END, assigned_class)
            return assigned_class

    def get_sentence_for_word(self, word, open_locally=settings["open_locally"], write_to_file=settings["write_to_file"]):
        """ TODO """
        """
            if you want to do fewer than 50 requests, you don’t need to use any waiting,
            if you want to do up to 900 requests, you need to use the interval of 4 seconds per query,
            if you want to do more than 2000 requests, you need to use interval ca 44 seconds.
        """

        if open_locally:
            with open(DATA_WSD_SENTENCES_PATH + word + '.json', 'r') as f:
                data = json.load(f)
                return self.parse_result(data)

        else:

            time.sleep(10)

            url = base_url + method
            attrs = dict(corpname=corpname, q='', pagesize='200', format='json', username=username, api_key=api_key, viewmode='sentence', lpos='-n', async=0, gdex_enabled=1)
            attrs['q'] = 'q' + '[lemma="'+word+'"]'
            headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
            r = s.get(url, params=attrs, headers=headers)

            if r.status_code == 429:
                print('Error: 429')
                return False

            else:
                json_response = r.json()

                if write_to_file:
                    with open(DATA_WSD_SENTENCES_PATH + word + '.json', 'w') as outfile:
                        json.dump(json_response, outfile)

                return self.parse_result(json_response)

    def parse_result(self, json_obj,
                     return_keyword=settings["return_keyword"],
                     return_single_sentence=settings["return_single_sentence"]):

        result_count = int(json_obj.get('concsize', '0'))
        text = ''
        if result_count > 0:
            response = json.dumps(json_obj["Lines"], sort_keys=True, indent=4, ensure_ascii=False)
            item_dict = json.loads(response)
            sentences_count = len(item_dict)
            if sentences_count > 0:
                counter = 0
                if return_single_sentence:
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

                    if return_keyword:
                        text += left + kwic + right + ' '
                    else:
                        text += left + right + ' '

                else:
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

                        if return_keyword:
                            text += left + kwic + right + ' '
                        else:
                            text += left + right + ' '

                        counter += 1

            return text

        else:
            return False

    def return_most_frequent_sense(self, word, n_dict, y_dict):
        """ This function returns the most frequents sense for a word, given two dictionaries with instances. """
        count_n = n_dict[word]
        count_y = y_dict[word]

        if count_y > count_n:
            return 1
        else:
            return 0


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
    PREF_WSD = Wsd('Prefixoids', DATA_WSD_PATH + PREF_JSON_DICT)
    pref_inventory_list = PREF_WSD.read_file_to_list(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE)
    n_pref_dict = PREF_WSD.create_affixoid_dictionary(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE, 'N')
    y_pref_dict = PREF_WSD.create_affixoid_dictionary(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE, 'Y')
    f0_pref_wsd_labels = []  # wsd labels
    f1_pref_wsd_list = []  # wsd predicitons

    n_text = ''
    y_text = ''

    dictionary_n = PREF_WSD.create_empty_dictionary(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE)
    dictionary_y = PREF_WSD.create_empty_dictionary(DATA_FINAL_PATH + FINAL_PREFIXOID_FILE)
    """ Loop """
    counter = 0
    for i in pref_inventory_list:
        counter += 1
        # started from 0 - 07.08. 12:25
        # stopped at 400 - 07.08. 16:57
        # stopped at 600 - 07.08. 18:04
        # stopped at 800 - 07.08. 19:06
        # stopped at 1000 - 07.08. 22:33
        # sentences start at 1000
        # stopped at 1100 - 08.08. 01:08
        # stopped at 1190 - 08.08. 12:03
        # stopped at 1999 - 08.08. 18:27
        # stopped at END 2009 - 08.08. 18:35
        # BEGIN stopped at 40 - 08.08. 18:51
        # stopped at 370 - 08.08. 20:35

        if counter < 1000:
            pass
        # elif counter < 370:
        #     pass
        else:
            print('Line:', str(counter) + ' ===============================', i[0], i[-1])
            f0 = PREF_WSD.transform_class_name_to_binary(i[-1])
            f1 = PREF_WSD.lesk(i[2], i[0], n_pref_dict, y_pref_dict)

            if f1 == -1:
                pass

            else:
                if i[-1] == 'N':
                    dictionary_n[i[2]].append(PREF_WSD.get_sentence_for_word(i[0]))

                if i[-1] == 'Y':
                    dictionary_y[i[2]].append(PREF_WSD.get_sentence_for_word(i[0]))

                f0_pref_wsd_labels.append(f0)
                f1_pref_wsd_list.append(f1)

    # PREF_WSD.write_list_to_file(f0_pref_wsd_labels, DATA_WSD_PATH + 'f0_pref_wsd_final.txt')
    # PREF_WSD.write_list_to_file(f1_pref_wsd_list, DATA_WSD_PATH + 'f1_pref_wsd_final.txt')

    print(dictionary_n.keys())
    print('===')
    print(dictionary_y.keys())

    f_y = open('yes.txt', 'w', encoding='utf-8')
    f_n = open('no.txt', 'w', encoding='utf-8')
    f_y.write(str(dictionary_y))
    f_n.write(str(dictionary_n))

    # """
    #     SUFFIXOIDS WSD
    # """
    # SUFF_WSD = Wsd('Suffixoids', DATA_WSD_PATH + SUFF_JSON_DICT)
    # suff_inventory_list = SUFF_WSD.read_file_to_list(DATA_FINAL_PATH + FINAL_SUFFIXOID_FILE)
    # f0_suff_wsd_labels = []  # wsd labels
    # f1_suff_wsd_list = []  # wsd predicitons

    """ Tests """
    # print(PREF_WSD.is_in_germanet('Test'))
    # print(PREF_WSD.search_germanet_synsets('Bilderbuch'))
    # print(PREF_WSD.create_synsets_dictionary('Schwein', return_wiktionary_sense=False))
    # print(PREF_WSD.calculate_similarity_scores('Bilderbuch', 'Auflage'))
    # print(PREF_WSD.get_sentence_for_word('Bilderbuchabsturz', open_locally=True))
    # print(PREF_WSD.lesk('Jahrhundert', 'Jahrhunderthälfte', n_pref_dict, y_pref_dict, lemmatize=True))

    inventory = ['Bilderbuch',
                 'Blitz',
                 'Bombe',
                 'Glanz',
                 'Heide',
                 'Jahrhundert',
                 'Qualität',
                 'Schwein',
                 'Spitze',
                 'Traum']

    # for i in inventory:
    #     print(i, PREF_WSD.return_most_frequent_sense(i, n_pref_dict, y_pref_dict))
