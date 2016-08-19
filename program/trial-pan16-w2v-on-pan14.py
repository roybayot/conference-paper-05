#!/usr/bin/env python
#!/home/darklord/anaconda2/bin/python

import sys
import getopt
import bleach
import xml.etree.ElementTree as ET
import os
import re
import pickle


import pandas as pd
import numpy as np
import re
import timeit
import gensim

from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from xml.etree.ElementTree import ParseError

from helpers import *


reload(sys)
sys.setdefaultencoding("ISO-8859-1")


if __name__ == "__main__":
    argv = sys.argv[1:]
    
    num_features = 100
    langs=["english", "spanish"]
    tasks=["age", "gender"]
    scoring_function = "accuracy"

    current_working_dir = './'
    model_dir = "word2vec-models/wikipedia-only-trained-on-my-machine/"

    classification_models = {}
    
    relations = {'dutch': {'truth_file': 'summary-dutch-truth.txt',\
                        'model_file': 'wiki.nl.tex.d100.model'
                        },
                'english': {'truth_file': 'summary-english-truth.txt',\
                            'model_file': 'wiki.en.tex.d100.model'
                            },
                'spanish': {'truth_file': 'summary-spanish-truth.txt',\
                            'model_file': 'wiki.es.tex.d100.model'
                            }
                }
    
    inputDir1, inputDir2, outputDir = getRelevantDirectories2(argv)

    allPaths = getAllFilenamesWithAbsPath(inputDir1)
    allTruthText_src = getTruthTextFiles(allPaths)

# remove dutch
    allTruthText_src = [f for f in allTruthText_src if "dutch" not in f]
#    allTruthText_src = [f for f in allTruthText_src if "spanish" not in f]

    allPaths = getAllFilenamesWithAbsPath(inputDir2)
    allTruthText_dst = getTruthTextFiles(allPaths)
    
    modelDir = outputDir 

    for one_truth_file in allTruthText_src:
        truth_file = one_truth_file
        train = pd.read_csv(truth_file, header=0, delimiter="\t", quoting=1)
        lang = getLang(one_truth_file, langs)
#        print lang

        model_file = current_working_dir + model_dir + relations[lang]['model_file']
        clean_train_data = makeLowerCase(train['text'])

        model = gensim.models.Word2Vec.load(model_file)
        trainDataVecs, trashedWords = getAvgFeatureVecs( clean_train_data,\
                                                         model,\
                                                         num_features )
        X = trainDataVecs



        all_poly_results = {}
        all_rbf_results = {}
        
        for task in tasks:
#            print task
            if lang == "dutch" and task == "age":
                print "nothing"
            else:
# train one                
                y = train[task]
# must put here to indicate what type of classifier was used
                clf = svm.SVC(kernel='rbf', gamma=100, C=100)
                clf.fit(X, y)
# get the files
                target_files = [f for f in allTruthText_dst if lang in f]
#                print target_files
# minimize target
#                target_files = [target_files[0]]
#                print target_files
                for one_target_file in target_files:
#                    print one_target_file
                    target = pd.read_csv(one_target_file, header=0, delimiter="\t", quoting=1)
                    clean_target_text = makeLowerCase(target['text'])

                    trainDataVecs, trashedWords = getAvgFeatureVecs( clean_target_text,\
                                                         model,\
                                                         num_features )
                    XX = trainDataVecs
                    yy = clf.predict(XX)
#                    print yy

# compare prediction and actual
                    actual = target[task]
                    zz = zip(actual, yy)
#                    print zz

                    match = [1 for x,y in zz if x == y]
                    accuracy = float(len(match))/len(yy)


                    print "="*80
                    print lang, task, "accuracy: ", accuracy
                    print "dataset source:", one_truth_file
                    print "dataset target:", one_target_file



