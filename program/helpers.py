import getopt
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import sys
import os

import sklearn
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from scipy import stats

import re

import xml.etree.ElementTree as ET
import sys
import re


reload(sys)
#sys.setdefaultencoding("ISO-8859-1")
sys.setdefaultencoding("UTF-8")

def getRelevantDirectories2(argv):
    inputDir1 = ''
    inputDir2 = ''
    outputDir = ''
    modelDir = ''

    try:
        opts, args = getopt.getopt(argv,"hab:o:",["ifile1=","ifile2=","ofile="])
        print opts
    except getopt.GetoptError:
        print './myTrainingSoftware.py -i <inputdirectory> -o <outputdirectory>'
        print 'The input directory should contain all the training files. \nThe output directory will be where the models are stored.'
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print './myTrainingSoftware.py -i <inputdirectory> -o <outputdirectory>'
            print 'The input directory should contain all the training files. \nThe output directory will be where the models are stored.'
            sys.exit()
        elif opt in ("-a", "--ifile1"):
            inputDir1 = arg
        elif opt in ("-b", "--ifile2"):
            inputDir2 = arg
        elif opt in ("-o", "--ofile"):
            outputDir = arg
    return inputDir1, inputDir2, outputDir



def getLang(truth_file, all_langs):
    a = truth_file.strip().split("/")
#    a = a[-1]
#    print a
    lang = [ lang for lang in all_langs if lang in truth_file]
    print lang
    print "Processing: ", lang[0]
    lang = lang[0]
    return lang

def makeLowerCase(clean_train_data):
    lower_case_clean_train_data = []
    for each_line in clean_train_data:
        try:
            lower_case_clean_train_data.append(each_line.lower())
        except:
            lower_case_clean_train_data.append('')

    return lower_case_clean_train_data


class MyXMLParser(ET.XMLParser):

    rx = re.compile("&#([0-9]+);|&#x([0-9a-fA-F]+);")

    def feed(self,data):
        mydata = data
        m = self.rx.search(data)
        if m is not None:
            target = m.group(1)
            if target:
                num = int(target)
            else:
                num = int(m.group(2), 16)
            if not(num in (0x9, 0xA, 0xD) or 0x20 <= num <= 0xD7FF
                   or 0xE000 <= num <= 0xFFFD or 0x10000 <= num <= 0x10FFFF):
                # is invalid xml character, cut it out of the stream
                print 'removing %s' % m.group()
                mstart, mend = m.span()
                mydata = data[:mstart] + data[mend:]
        else:
            mydata = data
        super(MyXMLParser,self).feed(mydata)

def clean_text(raw_text):
    review_text = BeautifulSoup(raw_text).get_text()
    words = review_text.lower().split()
    return(" ".join(words))

def clean_all_text(allText, numLines):
    clean_train_data = []
    for i in xrange(0, numLines):
        clean_train_data.append(clean_text(allText[i]))
    return clean_train_data

def getMeanAccuracies(results_dictionary):
    accuracies_dictionary = {}
    for each_key in results_dictionary.keys():
        list_of_accuracies = results_dictionary[each_key]
        accuracies = [a.mean() for a in list_of_accuracies]
        accuracies_dictionary[each_key] = accuracies
    return accuracies_dictionary

def getSortedKeys(all_results):
    sorted_keys = all_results.keys()
    sorted_keys.sort()
    return sorted_keys


def turnMeanAccuraciesToExcel(file_name, results_dictionary):
    accuracies_dictionary = getMeanAccuracies(results_dictionary)
    list_of_accuracy_settings = getListOfSettings(accuracies_dictionary)
    i = len(list_of_accuracies)
    accuracy_values = np.zeros(1,i)
    for x in range(i):
        accuracy_values[x] = accuracies_dictionary[list_of_accuracy_settings[x]]
    df = pd.DataFrame(data=accuracy_values, columns=list_of_accuracy_settings)
    df.index="accuracy"
    df.to_csv(file_name, sep=',', encoding='utf-8')
    
def makePValMatrix(all_results):
    sorted_keys = getSortedKeys(all_results)
    list_length = len(sorted_keys)
    p_value_matrix = np.zeros((list_length, list_length))
    i = range(0, list_length)
    #sig values
                            
    for key_1, x in zip(sorted_keys, i):
        for key_2, y in zip(sorted_keys, i):
            treatment_1 = all_results[key_1]
            treatment_2 = all_results[key_2]
            z_stat, p_val = stats.ranksums(treatment_1, treatment_2)
            p_value_matrix[x,y] = p_val
    
    return p_value_matrix

def turnPValMatrixToExcel(fileName, all_results):
    p_value_matrix = makePValMatrix(all_results)
    sorted_keys = getSortedKeys(all_results)
    df = pd.DataFrame(data = p_value_matrix, columns=sorted_keys)
    df.index = sorted_keys
    null_disproved = df[df < 0.05]
    null_disproved.to_csv(fileName, sep=',', encoding='utf-8')


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    wordsNotInDict = []

    try:
        words = words.split(" ")
    except AttributeError:
        featureVec = np.random.rand(num_features)
        return featureVec, wordsNotInDict

    
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
        else:
            wordsNotInDict.append(word)
            
    if nwords == 0.:
        featureVec = np.random.rand(num_features)
    else:
        featureVec = np.divide(featureVec,nwords)
    return featureVec, wordsNotInDict

def getAvgFeatureVecs(all_texts, model, num_features):
    counter = 0.
    reviewFeatureVecs = np.zeros((len(all_texts),num_features),dtype="float32")
    lineOfWordsNotInDict = []
    
    for one_line in all_texts:
        reviewFeatureVecs[counter], wordsNotInDict = makeFeatureVec(one_line, model, num_features)
        lineOfWordsNotInDict.append(wordsNotInDict)
        counter = counter + 1.
        
    return reviewFeatureVecs, lineOfWordsNotInDict


def doSVMwithPoly(trainDataVecs, targetVec, source, num_features, task,\
        num_folds=10, degrees=[1,2,3], C=[10**-1, 10, 10**3],\
        scoring_function="accuracy"):
    
    poly_results = {}
    for degree in degrees:
        for one_C in C:
            clf = svm.SVC(kernel='poly', degree=degree, coef0=one_C, gamma=1)
            scores = cross_validation.cross_val_score(clf, trainDataVecs,\
                                                      targetVec, cv=num_folds,\
                                                      scoring=scoring_function)

            string_pattern = "word2vec-source={} dims={} task={} kernel={} degree={} C={}"
                                                                        
            dict_key = string_pattern.format(source, num_features, task, \
                                             "poly", degree, one_C)
            poly_results[dict_key] = scores
    return poly_results


def doSVMwithRBF(trainDataVecs, targetVec, source, num_features, task,\
                 num_folds=10, gammas=[1, 0.001], C = [10, 1000],\
                 scoring_function="accuracy"):
   
    rbf_results = {}
    for g in gammas:
        for one_C in C:
            clf = svm.SVC(kernel='rbf', gamma=g, C=one_C)
            scores = cross_validation.cross_val_score(clf, trainDataVecs,\
                                                      targetVec, cv=10,\
                                                      scoring=scoring_function)
            
            string_pattern = "word2vec-source={} dims={} task={} kernel={} gamma={} C={}"
            dict_key = string_pattern.format(source, num_features, task, \
                                            "rbf",g, one_C)
            rbf_results[dict_key] = scores
    
    return rbf_results


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z



def dirExists(inputDir):
    if os.path.exists(inputDir):
        return True
    elif os.access(os.path.dirname(inputDir), os.W_OK):
        print "Cannot access the directory. Check for privileges."
        return False
    else:
        print "Directory does not exist."
        return False

def absoluteFilePaths(directory):
    allPaths = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            onePath = os.path.abspath(os.path.join(dirpath, f))
            allPaths.append(onePath)
    return allPaths

def getAllFilenamesWithAbsPath(inputDir):
    if dirExists(inputDir):
        allPaths = absoluteFilePaths(inputDir)
        return allPaths
    else:
        sys.exit()

def isTruthTextFile(f):
    return 'truth.txt' in f

def getTruthTextFiles(allPaths):
    return [f for f in allPaths if isTruthTextFile(f)]



def getRelevantDirectories(argv):
    inputDir = ''
    outputDir = ''
    modelDir = ''
    
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print './myTrainingSoftware.py -i <inputdirectory> -o <outputdirectory>'
        print 'The input directory should contain all the training files. \nThe output directory will be where the models are stored.'
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print './myTrainingSoftware.py -i <inputdirectory> -o <outputdirectory>'
            print 'The input directory should contain all the training files. \nThe output directory will be where the models are stored.'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputDir = arg
        elif opt in ("-o", "--ofile"):
            outputDir = arg
    return inputDir, outputDir


def writeOneSummary(outputFilename, oneTruthFile, allPaths, realOutputFilename):
    data = ["filename", "gender", "age", "text"]
    path = outputFilename.strip().split("/")
    outputFilename = path[-1]
    print "Output filename: ", outputFilename

    path = '/'.join(path[0:-1])

    tsv_writer(data, realOutputFilename)

    current_edition = "pan16"
    
    if current_edition == "pan14" or current_edition == "pan16":
    	gender = {'MALE': 0, 'FEMALE':1}
    	ageGroup = {'18-24': 0, \
                	'25-34': 1, \
                	'35-49': 2, \
                	'50-64': 3, \
                	'65-xx': 4, \
                	'XX-XX': None}
    


    if current_edition == "pan15":
    	gender 	 = {'M': 0, 'F':1}
    	ageGroup = {'18-24': 0, \
                	'25-34': 1, \
                	'35-49': 2, \
                	'50-XX': 4, \
                	'XX-XX': None}




    one_file = open(oneTruthFile, 'r')

    for line in one_file:
        a = line.strip().split(":::")
        fileName 		  = path+ "/" + a[0] + ".xml"
# 		print fileName
        thisGender 	 	  = gender[a[1]]
        thisAgeGroup 	  = ageGroup[a[2]]
        
        parser = helpers.MyXMLParser(encoding='utf-8')

        print "opening:", fileName

        try:
            tree = ET.parse(fileName, parser=parser)
            #tree = ET.parse(fileName)
            #print "Filename: %s SUCCESS!" % fileName
        except ParseError:
            with open(fileName, 'r') as f:
                read_data = f.read()
                read_data = read_data.replace("&#11;", "")
            
            with open('temp_file.xml', 'w') as g:
                g.write(read_data)
            tree = ET.parse('temp_file.xml')
        except:
            e = sys.exc_info()[0]
            print "Filename: %s Error: %s" % (fileName, e)
        else:
            root = tree.getroot()
            a = []
            for x in root.iter("document"):
                a.append(x.text)

# 			print "In Else"
            allText = ""

# 			print "Going in for loop"
           
            for doc in a:
                clean = bleach.clean(doc, tags=[], strip=True)
                allText = allText + clean
            #allText = allText.encode('ISO-8859-1')
            allText = allText.encode('utf-8')
            #allText = allText.encode('latin-1')
            #allText = allText.replace("\"", " ")
            #allText = allText.replace("...", " ")
# 			print "Out of loop, writing"								
            data = [fileName, thisGender, thisAgeGroup, allText]
            tsv_writer(data, realOutputFilename)
            print "done with file"
# 			print "Finish writing one line"

def tsv_writer(data, path):
    """ Write data to a TSV file path """
    with open(path, "a") as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        writer.writerow(data)


def deriveOutputFilename(oneTruthFile, langs):
    a = oneTruthFile.strip().split("/")
    lang = [ lang for lang in langs if lang in oneTruthFile]
    print "Processing: ", lang[0]

    outputFilename = '/'.join(a[0:-1]) + '/summary-' + lang[0] + '-' + a[-1]
    #outputFilename = 'summary-' + lang[0] + '-' + a[-1]
    return outputFilename

def generateTruthTexts(allPaths, allTruthText, outputDir, langs):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    for oneTruthFile in allTruthText:
        outputFilename = deriveOutputFilename(oneTruthFile, langs)

        some_fill_in = outputFilename.split("/")

        print some_fill_in[-1]
        #realOutputFilename = os.getcwd() + '/' + outputDir + '/' + some_fill_in[-1] 
        realOutputFilename = outputDir + '/' + some_fill_in[-1] 
        print outputFilename
        writeOneSummary(outputFilename, oneTruthFile, allPaths, realOutputFilename)

def trainOne(X,y,lang,task):
    if lang == "english" and task == "age":
        clf = svm.SVC(kernel='rbf', gamma=100, C=100)
        clf.fit(X, y)
    if lang == "english" and task == "gender":
        clf = svm.SVC(kernel='rbf', gamma=100, C=100)
        clf.fit(X, y)
    if lang == "dutch" and task == "age":
        clf = [] 
    if lang == "dutch" and task == "gender":
        clf = svm.SVC(kernel='rbf', gamma=1, C=100)
        clf.fit(X, y)
    if lang == "spanish" and task == "age":
        clf = svm.SVC(kernel='rbf', gamma=1, C=100)
        clf.fit(X, y)
    if lang == "spanish" and task == "gender":
        clf = svm.SVC(kernel='rbf', gamma=1, C=100)
        clf.fit(X, y)
    
    return clf

def writeModels(models, outputDir):
    fileName = outputDir + "/models.pkl"
    f = open(fileName, 'wb')
    pickle.dump(models, f)
    f.close()
