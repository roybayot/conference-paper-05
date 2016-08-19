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

    modelDir = outputDir 
#    import ipdb; ipdb.set_trace()

    for one_truth_file in allTruthText:
        train = pd.read_csv(one_truth_file, header=0, delimiter="\t", quoting=1)
        lang = getLang(one_truth_file, langs)

        print lang
        
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
            if lang == "dutch" and task == "age":
                print "nothing"
            else:
                y = train[task]
                poly_results = doSVMwithPoly(X,y,"tfidf-pan15", 10000, task, 10,\
                                                    degrees=[1,2,3],\
                                                    C=[10**-1, 10, 1**3])
                rbf_results = doSVMwithRBF(X,y, "tfidf-pan15", 10000, task, 10,\
                                                    gammas=[0.001, 1, 100], \
                                                    C=[1, 10, 1000])

                all_poly_results = merge_two_dicts(all_poly_results, \
                                                        poly_results)
                all_rbf_results = merge_two_dicts(all_rbf_results, \
                                                        rbf_results)
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

    all_poly_results = {}
    all_rbf_results = {}

    return all_poly_results, all_rbf_results
    
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

