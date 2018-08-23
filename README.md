# Bachelorarbeit 
Collection of modules for feature extraction, Lesk algorithm and classification for the bachelor thesis titled "Disambiguating nominal prefixoid and suffixoid formations from compounds"

## Installation
Requirements
- [python3](https://www.python.org/downloads/)
- [GermaNet](http://www.sfs.uni-tuebingen.de/GermaNet/)
- [lxml](https://lxml.de/)
- [matplotlib](https://matplotlib.org/)
- [nltk](https://www.nltk.org/)
- [numpy](http://www.numpy.org/)
- [MongoDB](https://www.mongodb.com/)
- [pygermanet](https://github.com/wroberts/pygermanet)
- [pymongo](http://api.mongodb.com/python/current/)
- [requests](http://docs.python-requests.org/en/master/)
- [scipy](https://www.scipy.org/)
- [sklearn](http://scikit-learn.org/stable/)

Clone the repository
```bash
$ git clone https://gitlab.cl.uni-heidelberg.de/spahic/ba-ss18.git
```

Install requirements
```bash
$ pip install -r src/requirements.txt
```

## Project structure
```
ba-ss18/
├── data
│   ├── features
│   ├── final
│   ├── statistics
│   └── wsd
│       └── sdewac2
│           ├── final
│           └── sentences
├── res
│   ├── AffectiveNorms
│   ├── EmoLex
│   ├── fastText
│   ├── GermaNet
│   ├── PMI
│   ├── PolArtUZH
│   ├── SentiMerge
│   └── SentiWS
└── src
    └── modules
        └── doctests
```

## Configuration file
`config.ini` contains all the required settings for filenames and paths. The main modules `FeatureExtractor.py`, `Wsd.py` and `Classifier.py` are pre-configured to use files from the resources.

```bash
$ src/config.ini
```

## Feature Extraction for word embeddings
Before extraction, set correct path to the file with normalized pmi values (due to size not in this package) `sdewac_npmi.csv.bz2` in:
```bash
$ src/config.ini
```

and run:
```bash
$ python src/FeatureExtractor.py
```

## Word Sense Disambiguation
Before running the Word Sense Disambiguation module make sure that the GermaNet resource (available from the University of Tübingen) and `pygermanet` are installed. More info [here](https://github.com/wroberts/pygermanet)
```bash
$ python src/Wsd.py
```

## Classification and results
Results from `FeatureExtractor.py` and `Wsd.py` will be written to files set in `config.ini`. Default values are `ba-ss18/data/features/` and `ba-ss18/data/wsd/`. `Classifier.py` does not need GermaNet or PMI values, since the values are all available in the `features` folder. `Classifier.py` will load these features automatically and print results together with the most frequent sense baseline.
```bash
$ python src/Classifier.py
```

## Contributors
[Darmin Spahic](https://github.com/darminspahic)

## References
Agirre, Eneko, and Philip Edmonds. 2006. Word Sense Disambiguation: Algorithms and Applications (Text, Speech and Language Technology). Berlin, Heidelberg: Springer-Verlag.

Elsen, Hilke. 2014. Grundzüge Der Morphologie Des Deutschen. 2., aktualisierte Aufl. De Gruyter Studium. Berlin [u.a.]: De Gruyter.

Hamp, Birgit, and Helmut Feldweg. 1997. “GermaNet - a Lexical-Semantic Net for German.” Proceedings of the ACL Workshop Automatic Information Extraction and Building of Lexical Semantic Resources for NLP Applications.

Henrich, Verena, and Erhard Hinrichs. 2010. “GernEdiT - the Germanet Editing Tool.” Proceedings of the Seventh Conference on International Language Resources and Evaluation (LREC 2010), May. Valletta, Malta, 2228–35. http://www.lrec-conf.org/proceedings/lrec2010/pdf/264_Paper.pdf.

Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, et al. 2011. “Scikit-Learn: Machine Learning in Python.” Journal of Machine Learning Research 12: 2825–30.

Ruppenhofer, Josef, Petra Steiner, and Michael Wiegand. 2017. “Evaluating the Morphological Compositionality of Polarity,” November, 625–33.

Vapnik, Vladimir N. 1995. The Nature of Statistical Learning Theory. New York ; Berlin ; Heidelberg: Springer.

## License
Copyright 2018 [Darmin Spahic](https://github.com/darminspahic)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.