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


def main(argv):
    langs=["english", "dutch", "spanish"]
    tasks=["age", "gender"]
    scoring_function = "accuracy"

    current_working_dir = './'
    classification_models = {}
    
    inputDir, outputDir = getRelevantDirectories(argv)

    allPaths = getAllFilenamesWithAbsPath(inputDir)
    allTruthText = getTruthTextFiles(allPaths)

    modelDir = outputDir 

    for one_truth_file in allTruthText:
        truth_file = one_truth_file
        train = pd.read_csv(truth_file, header=0, delimiter="\t", quoting=1)
        lang = getLang(one_truth_file, langs)
        print lang

        clean_train_data = makeLowerCase(train['text'])

        tfidf_vectorizer = TfidfVectorizer(analyzer = "word",\
                                           tokenizer = None,\
                                           preprocessor = None,\
                                           stop_words = None)
        X = tfidf_vectorizer.fit_transform(clean_train_data)
        X = X.toarray()

        all_poly_results = {}
        all_rbf_results = {}
        
        for task in tasks:
            if lang == "dutch" and task == "age":
                print "nothing"
            else:
                y = train[task]
                poly_results = doSVMwithPoly(X,y,"tfidf-pan16", 10000, task, 10,\
                                                degrees=[1,2,3],C=[10**-1, 10, 1**3])
                rbf_results = doSVMwithRBF(X,y, "tfidf-pan16", 10000, task, 10,\
                                            gammas=[0.001, 1, 100], C=[1, 10, 1000])
                all_poly_results = merge_two_dicts(all_poly_results, poly_results)
                all_rbf_results = merge_two_dicts(all_rbf_results, rbf_results)
                
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

#        for task in tasks:
#        	if not (lang == "dutch" and task == "age"):
#        		y = train[task]
#        		import ipdb; ipdb.set_trace()
#        		poly_results 	= doSVMwithPoly(X,y,"tfidf-pan16", 10000, task, 10,\
#                                         	 degrees=[1,2,3],C=[10**-1, 10, 1**3])
#            	rbf_results 	= doSVMwithRBF(X,y, "tfidf-pan16", 10000, task, 10,\
#                                       	   gammas=[0.001, 1, 100], C=[1, 10, 1000])
#            	all_poly_results = merge_two_dicts(all_poly_results, poly_results)
#            	all_rbf_results = merge_two_dicts(all_rbf_results, rbf_results)
#
#        	
#        	if lang == "dutch" and task == "age":
#        		all_poly_results = {}
#        		all_rbf_results = {}
#        	elif lang == "dutch" and task == "gender":
#        		y = train[task]
#        		import ipdb; ipdb.set_trace()
#        	else:
#        		y = train[task]
#        		import ipdb; ipdb.set_trace()
#        		poly_results 	= doSVMwithPoly(X,y,"tfidf-pan16", 10000, task, 10,\
#                                         	 degrees=[1,2,3],C=[10**-1, 10, 1**3])
#            	rbf_results 	= doSVMwithRBF(X,y, "tfidf-pan16", 10000, task, 10,\
#                                       	   gammas=[0.001, 1, 100], C=[1, 10, 1000])
#            	all_poly_results = merge_two_dicts(all_poly_results, poly_results)
#            	all_rbf_results = merge_two_dicts(all_rbf_results, rbf_results)
#        return all_poly_results, all_rbf_results


        

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

