import nltk
from nltk.corpus import stopwords
from nltk.tokenize import load
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem.snowball import GermanStemmer
import json

# Sentence tokenizer
sent_tok = load('tokenizers/punkt/german.pickle')

# Filter stopwords
# stop_words = set(stopwords.words('german'))
stop_words_german = stopwords.words('german')
stop_words_capitalize = [x.capitalize() for x in stop_words_german]
stop_words = set(stop_words_german + stop_words_capitalize)
print(stop_words)

# Word Tokenizer
word_tok2 = TreebankWordTokenizer()
word_tok = RegexpTokenizer(r'\w+', gaps=False, discard_empty=False)
stemmer = GermanStemmer()
# word_tok = TreebankWordTokenizer()

text = "Sie wollen auf keinen Fall abseits stehen , wenn die USA in Zentralasien die erste Front in ihrem Jahrhundert-Feldzug eröffnen 3. Mai 2014 und Dr.Meier . Heute ist der 3. Mai 2014 und Dr-Meier feiert heute seinen 43. Geburtstag. Das muss unbedingt heute daran denken, Mehl, usw. für einen Kuchen einzukaufen. Aber leider habe ich nur noch EUR 3.50 in meiner Brieftasche."
text = "der Zeitraum von hundert Jahren, vom 1. Januar des xxx1. Jahres bis zum 31. Dezember des x100. Jahres; Beginn der Zählung in die positive Richtung am 1. 1. des Jahres 1 (n. Chr. oder u. Z.); Beginn der Zählung in die negative Richtung am 31. 12. des Jahres 1 (v. Chr. oder v. u. Z.) bzw. am 1. 1 des Jahres x100 bis zum 31. 12. des Jahres xxx1."

sents = sent_tok.tokenize(text)
words = word_tok.tokenize(text)
words2 = word_tok2.tokenize(text)
# stemmed = stemmer.stem(words)

clean = [w for w in word_tok.tokenize(text) if w not in stop_words]
clean2 = [w for w in words if w not in stop_words]

# print(sents)
# print(words)
# print(clean)
print(set(clean))
print(set(clean2))
# print(stemmed)

print()

from nltk.corpus import wordnet
s = wordnet.synsets('computer')
print(s)
for i in s:
    print(i.definition())

print()

from nltk.wsd import lesk
sent = ['I', 'went', 'financial', 'account', 'savings',  'to', 'the', 'bank', 'to', 'deposit', 'money', '.']

print(lesk(sent, 'bank'))
print(lesk(sent, 'bank').definition())


print("=========")
g = {'Glanz': [{1: 'übertragen: besonderer, auffälliger Zustand'}, {2: 'Schein oder Widerschein, besonders auf glatten Materialien; das Leuchten von etwas'}]}
a = ['Arbeit', {'3': 'Ergebnis einer Tätigkeit; Produkt, Werk'}, {'4': 'auszuführende, zweckgerichtete Tätigkeit'}, {'5': 'Verhältnis, bei dem man eine Tätigkeit gegen Geld verrichtet'}, {'1': None}, {'2': 'Physik: Energie, die durch Kraft über einen Weg auf einen Körper übertragen wird'}]

print(g['Glanz'])

g1='übertragen: besonderer, auffälliger Zustand'
g2='Schein oder Widerschein, besonders auf glatten Materialien; das Leuchten von etwas'

a1= 'kleine, geheftete, nicht gebundene Schrift mit wenigen Seiten'

g1words = word_tok.tokenize(g1)
g2words = word_tok.tokenize(g2)
a1words = word_tok.tokenize(a1)


g1clean = [w for w in g1words if w not in stop_words]
g2clean = [w for w in g2words if w not in stop_words]
a1clean = [w for w in a1words if w not in stop_words]

print(set(g1clean))
print(set(g2clean))
print(set(a1clean))

contextg1 = set(g1words)
contextg2 = set(g2words)
contexta1 = set(a1words)

print(contextg1.intersection(contexta1))
print(contextg2.intersection(contexta1))

print()

json_file = open('../../data/wsd/pref_dictionary.json', 'r', encoding='utf-8')
json_data = json.load(json_file)

# print([w for w in word_tok.tokenize(json_data['Apostel']['0']['Duden']['Bedeutung']) if w not in stop_words])
# print([w for w in word_tok.tokenize(json_data['Apostel']['1']['Duden']['Bedeutung']) if w not in stop_words])
# print(set([w for w in word_tok.tokenize("drückt in Bildungen mit Substantiven aus, dass jemand oder etwas als ideal angesehen wird; drückt in Bildungen mit Substantiven aus, dass jemand oder etwas nur im Bilderbuch, aber nicht in der Realität existiert") if w not in stop_words]))

h = 'persönliches Notizbuch, in dem Tagesereignisse festgehalten werden.'

apo = json_data['Traum']['0']['Duden']['Bedeutung'] + ' ' + json_data['Traum']['0']['GermaNet']['Bedeutung'] + ' ' + json_data['Traum']['0']['Wiktionary']['Bedeutung']
apos = json_data['Traum']['1']['Duden']['Bedeutung'] + ' ' + json_data['Traum']['1']['GermaNet']['Bedeutung'] + ' ' + json_data['Traum']['1']['Wiktionary']['Bedeutung']

apob = json_data['Traum']['0']['Duden']['Beispiel'] + ' ' + json_data['Traum']['0']['GermaNet']['Beispiel'] + ' ' + json_data['Traum']['0']['Wiktionary']['Beispiel']
aposb = json_data['Traum']['1']['Duden']['Beispiel'] + ' ' + json_data['Traum']['1']['GermaNet']['Beispiel'] + ' ' + json_data['Traum']['1']['Wiktionary']['Beispiel']

s1 = set([w for w in word_tok.tokenize(apo + ' ' + apob) if w not in stop_words])
s2 = set([w for w in word_tok.tokenize(apos + ' ' + aposb) if w not in stop_words])

print([s for s in s1 if s not in s2])
print(s1.intersection(s2))

print(set([w for w in word_tok.tokenize(apos + ' ' + aposb)]).intersection(set([w for w in word_tok.tokenize(h)])))

bank = 'a financial institution that accepts deposits and channels the money into lending activities'
sent = 'The bank profits from the difference between the level of interest it pays for deposits, and the level of interest it charges in its lending activities.'

print(set([w for w in word_tok.tokenize(bank) if w not in set(stopwords.words('english'))]).intersection(set([w for w in word_tok.tokenize(sent) if w not in set(stopwords.words('english'))])))
