#!/usr/bin/env python
#!/home/darklord/anaconda2/bin/python

import sys
import getopt
import bleach
import xml.etree.ElementTree as ET
import os
import re
import csv
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




def main(argv):
    num_features = 100
    langs=["english", "dutch", "spanish"]
    tasks=["age", "gender"]
    scoring_function = "accuracy"
#    current_working_dir = os.getcwd() + '/'
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
    inputDir, outputDir = getRelevantDirectories(argv)
    print "INPUT DIR", inputDir
    print "OUTPUT DIR", outputDir 
    allPaths = getAllFilenamesWithAbsPath(inputDir)
    allTruthText = getTruthTextFiles(allPaths)
    print "ALL TRUTH TEXT", allTruthText
    generateTruthTexts(allPaths, allTruthText, outputDir, langs)

    #modelDir = os.getcwd() + '/' + outputDir 
    modelDir = outputDir 

    for f in allTruthText:
        a = f.strip().split("/")
        lang = [ lang for lang in langs if lang in f]
        print "Processing: ", lang[0]
        lang = lang[0]

        truth_file = relations[lang]['truth_file']
        model_file = current_working_dir + model_dir + relations[lang]['model_file']

        #truth_file = current_working_dir + outputDir + "/" + truth_file
        truth_file = outputDir + "/" + truth_file
        train = pd.read_csv(truth_file, header=0, delimiter="\t", quoting=1)
        print "Done reading file"
        max_n_words = 10000
        clean_train_data = train['text']

        tfidf_vectorizer = TfidfVectorizer(analyzer = "word",\
                                           tokenizer = None,\
                                           preprocessor = None,\
                                           stop_words = None,
                                           max_features = max_n_words)
        X = tfidf_vectorizer.fit_transform(clean_train_data)
        X = X.toarray()

        all_poly_results = {}
        all_rbf_results = {}

        for task in ["gender"]:
            y = train[task]
            poly_results = helpers.doSVMwithPoly(X,y,"tfidf-pan15", 10000, task, 10,\
                                                 degrees=[1,2,3],\
                                                 C=[10**-1, 10, 1**3])
            rbf_results = helpers.doSVMwithRBF(X,y, "tfidf-pan15", 10000, task, 10,\
                                                 gammas=[0.001, 1, 100], \
                                                 C=[1, 10, 1000])

            all_poly_results = helpers.merge_two_dicts(all_poly_results, \
                                                       poly_results)
            all_rbf_results = helpers.merge_two_dicts(all_rbf_results, \
                                                      rbf_results)

        return all_poly_results, all_rbf_results









        
#        model = gensim.models.Word2Vec.load(model_file)
#        
#        trainDataVecs, trashedWords = getAvgFeatureVecs( clean_train_data,\
#                                                         model,\
#                                                         num_features )
#        X = trainDataVecs
#
#        for task in tasks:
#            key_name = lang + "_" + task
#            
#            if lang == "dutch" and task == "age":
#                classification_models[key_name] = []
#            else:
#                y = train[task]
#                one_model = trainOne(X, y, lang, task)
#                classification_models[key_name] = one_model
#
#    writeModels(classification_models, modelDir)

if __name__ == "__main__":
    all_poly_results, all_rbf_results = main(sys.argv[1:])

    poly_keys = all_poly_results.keys()
    rbf_keys = all_rbf_results.keys()

    poly_keys.sort()
    rbf_keys.sort()

    print "="*80
    print "POLYNOMIAL"
    print "="*80

    for one_key in poly_keys:
        one_mean = all_poly_results[one_key].mean()
        print one_key, ": ", one_mean
        
    print "="*80
    print "RADIAL BASIS FUNCTION"
    print "="*80

    for one_key in rbf_keys:
        one_mean = all_rbf_results[one_key].mean()
        print one_key, ": ", one_mean

