""" pathutils.py
pathutils.py is a set of functions and definitions to support the programs associated with
NSDUH pathway analysis.
@author  Matthew J. Beattie
@date    December 20, 2020
"""

"""
Import python modules
"""
import pandas as pd
import numpy as np
import copy
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import random
import math as math
import json
import itertools

# SciKit library import
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import explained_variance_score, confusion_matrix, balanced_accuracy_score
from sklearn.metrics import f1_score, mean_squared_error, roc_curve, auc, average_precision_score
from sklearn.metrics import precision_recall_curve

""" DataFrameSelector()
Inputs:
attribs - a list of columns to extract

Output:
A pandas dataframe with only the selected columns
"""


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribs):
        self.attribs = attribs

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X[self.attribs]


""" DataSampleDropper()
Output:
A numpy array with no missing values
"""


class DataSampleDropper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X.dropna(how='any')


""" DropAges() 
Drops all rows of a pandas dataframe for which the age category is equal to a passed value
"""


class DropAges(BaseEstimator, TransformerMixin):
    def __init__(self, attribs):
        self.attribs = attribs

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        for i in self.attribs:
            querystr = 'CATAG6 != ' + str(i)
            X = X.query(querystr)
        return X


""" MakeNeedle() 
Takes the ANYNEEDL and NEDHER variables and creates a new variable that is 1
when a respondent has used needles to do drugs other than heroin
"""


class MakeNeedle(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X['NEEDNOHER'] = np.where((X.ANYNEEDL == 1) & (X.NEDHER == 0), 1, 0)
        X2 = X.drop(['ANYNEEDL', 'NEDHER'], axis=1)
        return X2


""" SvcFlag() 
Sets a flag for existing or prior military service
"""


class SvcFlag(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X['SVCFLAG'] = np.where((X.SERVICE == 1), 1, 0)
        X2 = X.drop(['SERVICE'], axis=1)
        return X2


""" AFUSet() 
Reset flag values for AFU for pain relievers, stimulants, tranquilizers,
and sedatives.  These variables are different than normal drugs in NSDUH.  The AFU values are only collected
for respondents who initiated use in the last 12 months.  Users of these drugs who are not new initiates are
coded as '993'.  We set these users to NaN for imputation.  True missings are set to '991' to match the other
variables.
"""


class AFUSet(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X['IRPNRNMAGE'] = np.where((X.IRPNRNMAGE == 999), 991, np.where((X.IRPNRNMAGE == 993), None, X.IRPNRNMAGE))
        X['IRTRQNMAGE'] = np.where((X.IRTRQNMAGE == 999), 991, np.where((X.IRTRQNMAGE == 993), None, X.IRTRQNMAGE))
        X['IRSTMNMAGE'] = np.where((X.IRSTMNMAGE == 999), 991, np.where((X.IRSTMNMAGE == 993), None, X.IRSTMNMAGE))
        X['IRSEDNMAGE'] = np.where((X.IRSEDNMAGE == 999), 991, np.where((X.IRSEDNMAGE == 993), None, X.IRSEDNMAGE))
        return X


""" AFUGroup() 
Set categorical AFU flags for sets of drugs.  Cocaine and crack cocaine are combined into one category.  
Prescription drugs, represented by pain relievers, stimulants, and tranquilizers are grouped into
one category as well.
"""


class AFUGroup(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X['IRTOBAGE'] = X.loc[:, ['IRCIGAGE','IRCGRAGE','IRSMKLSSTRY']].min(axis=1)
        X['IRCOC2AGE'] = X.loc[:, ['IRCOCAGE','IRCRKAGE']].min(axis=1)
        X.drop(['IRCIGAGE','IRCGRAGE','IRSMKLSSTRY','IRCOCAGE', 'IRCRKAGE'], axis=1, inplace=True)
        return X


""" MakeInt32() 
Converts a set of given columns of a pandas dataframe to 32 bit integers
"""


class MakeInt32(BaseEstimator, TransformerMixin):
    def __init__(self, attribs):
        self.attribs = attribs

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        for attrib in self.attribs:
            X[attrib] = X[attrib].astype('int32')
        return X


""" MakeCat() 
Converts a set of given columns in a pandas dataframe to categorical variables
"""


class MakeCat(BaseEstimator, TransformerMixin):
    def __init__(self, attribs):
        self.attribs = attribs

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        for attrib in self.attribs:
            X[attrib] = X[attrib].astype('category')
        return X


""" MakeAllCat() 
Converts all columns in a pandas dataframe to categorical variables
"""


class MakeAllCat(BaseEstimator, TransformerMixin):
    def __init__(self, attribs):
        self.attribs = attribs

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X = X.astype('category')
        return X


""" CollapseMissing()
Collapses multiple values for skipped or missing values into one
"""


class CollapseMissing(BaseEstimator, TransformerMixin):
    def __init__(self, attribs, lowval, tgtval):
        self.attribs = attribs
        self.lowval = lowval
        self.tgtval = tgtval

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        for attrib in self.attribs:
            X.loc[X[attrib] > self.lowval, attrib] = self.tgtval
        return X


""" FillCatNAs()
Fill categorical NA values with a target value.
"""


class FillCatNAs(BaseEstimator, TransformerMixin):
    def __init__(self, attribs, replval):
        self.attribs = attribs
        self.replval = replval

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        for attrib in self.attribs:
            X[attrib] = X[attrib].cat.add_categories(self.replval)
            X[attrib] = X[attrib].fillna(self.replval)
        return X


"""newid()
Generates a new respondent ID value from a concatenation of the year and raw ID
"""


def newid(df, attrib, year):
    return str(year) + str(df[attrib])


""" AddYearRespID()
Calls newid() and applies the value to a new respondent ID column.  Drops the old one.
"""


class AddYearRespID(BaseEstimator, TransformerMixin):
    def __init__(self, year):
        self.year = year

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X['RESPID'] = X.apply(lambda row: newid(row, 'QUESTID2', self.year), axis=1)
        X['RESPID'] = X['RESPID'].astype('string')
        X2 = X.drop(['QUESTID2'], axis=1)
        return X2


""" makedict()
Creates a path of tuples sorted by age of first use.  Breaks ties randomly by adding a random number along
uniform(0,1) to every element.  Note:  Some variables use 991 for no use, some use 999
"""


def makedict(df, fieldlist, namedict):
    vect = []
    vect.append(('start',0))
    for field in fieldlist:
        if df[field] < 991:
            vect.append((namedict[field], df[field] + random.uniform(0, 1)))
    vect.sort(key=lambda x: x[1])
    return vect


""" MakePath()
Creates a column representing the path of drugs by AFU value.  Calls makedict().
"""


class MakePath(BaseEstimator, TransformerMixin):
    def __init__(self, afuvals, drugnames):
        self.afuvals = afuvals
        self.drugnames = drugnames

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X['AFUPATH'] = X.apply(lambda row: makedict(row, self.afuvals, self.drugnames), axis=1)
        return X
    
    
""" makeshortvect()
Creates a vector of integers where each position represents a drug and its entry is the AFU
value.  Ties are allowed.  Leave no-drug used AFU at 991.
"""


def makeshortvect(df, afuvals, drugorder):
    vect = []
    for i in range(len(drugorder)):
        vect.append(0)
    for field in afuvals:
        if df[field] != 991:
            vect[drugorder[field]] = df[field]
        else:
            vect[drugorder[field]] = 991
    return vect


""" MakeVect()
Creates a vector where each position is a drug and the entry is the AFU value.  Calls makeshortvect().
"""


class MakeVect(BaseEstimator, TransformerMixin):
    def __init__(self, afuvals, drugorder):
        self.afuvals = afuvals
        self.drugorder = drugorder

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X['AFUVECT'] = X.apply(lambda row: makeshortvect(row, self.afuvals, self.drugorder), axis=1)
        return X


""" pathgraph()
Creates a weighted graph from a sequence of (drug,age) tuples.
"""


def pathgraph(df, field, drugnums):
    pathlist = df[field]
    pathwayGraph = np.zeros((len(drugnums), len(drugnums)))
    origin = 0
    cumweight = 0
    for item in pathlist:
        dest = drugnums[item[0]]
        weight = item[1] - cumweight
        pathwayGraph[origin][dest] = weight
        origin = dest
        cumweight += weight
    return pathwayGraph


""" MakeGraph()
Creates a weighted graph representing the transition through the drug pathway.
"""


class MakeGraph(BaseEstimator, TransformerMixin):
    def __init__(self, field, drugnums):
        self.field = field
        self.drugnums = drugnums

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X['PATHGRAPH'] = X.apply(lambda row: pathgraph(row, self.field, self.drugnums), axis=1)
        return X


""" uncodepath()
This routine takes an n x n array of paths and decodes it into a path of
drug use strings.
"""


#def uncodepath(pathmatrix: np.array, namedict):
#    drugnums, drugnames = surveyvars(2016)
#    rows = pathmatrix.shape[0]
#    cols = pathmatrix.shape[1]
#    pathlist = []
#    origin = 0
#    dest = 0
#    while dest != 9999:
#        dest = 9999
#        for j in range(0, cols):
#            if pathmatrix[origin][j] > 1e-5:
#                dest = j
#                pathlist.append(((drugnames[origin], drugnames[dest]), pathmatrix[origin][j]))
#        origin = dest
#    return pathlist


""" uncodevect()
This routine takes an 1 x n path array of paths and decodes it into a path of
drug use strings.  Values of 991 aren't added to the path.
"""

#  Note:  'year' has been hard coded as 2016
def uncodevect(pathvect: np.array, namedict):
    ident, rawafuvals, afuvals, drugnames, drugorder, drugnums, drugposition, startdemog, demographics = surveyvars(2016)
    pathlist = []
    for i in range(0,len(pathvect)):
        if pathvect[i] != 991:
            pathlist.append((drugposition[i], pathvect[i]))
    return (sorted(pathlist, key = lambda x: x[1]))


""" makePathVect()
Creates a field where the pathways n x n graph is converted into a pathways 1 x (nxn) vector.
Applied to weighted pathways data.
"""


def makePathVect(pathgraph: np.array):
    tmpgraph = copy.deepcopy(pathgraph)
    rows, cols = tmpgraph.shape
    return tmpgraph.reshape(1, rows * cols)[0]


""" makeUnwPathVect()
Creates a field where the pathways n x n graph is converted into a pathways 1 x (nxn) vector.
Applied to unweighted pathways data.
"""


def makeUnwPathVect(pathgraph: np.array):
    tmpgraph = copy.deepcopy(pathgraph)
    rows, cols = tmpgraph.shape
    for i in range(rows):
        for j in range(cols):
            if tmpgraph[i][j] > 1e-5:
                tmpgraph[i][j] = 1
    return tmpgraph.reshape(1, rows * cols)[0]


""" MakeCat() 
Converts a set of given columns in a pandas dataframe to categorical variables
"""


class MakeCat(BaseEstimator, TransformerMixin):
    def __init__(self, attribs):
        self.attribs = attribs

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        for attrib in self.attribs:
            X[attrib] = X[attrib].astype('category')
        return X


""" pathsum()
Adds the weights of a pathway vector to determine the final age at the end of the path
"""


def pathsum(df, vect):
    return sum(df[vect])


""" resultsOut(df,filename)
Converts variables to categorical and print out crosstabs and Chi Square analyses for
each demographic variable.  The prints out the medoid pathways.
Inputs:  df is the labelled demographic dataframe, filename is the file to write output to, model is the cluster model
Outputs:  Standard out is written to filename, and the uncoded dataframe is returned to the caller.
"""


def resultsOut(stryear, df, filename, model, demogs, jsonfile):
    # Get json file with variable definitions
    f1 = open(jsonfile, 'r')
    nsduhDecoder = json.load(f1)
    f1.close()

    ident, rawafuvals, afuvals, drugnames, drugorder, drugnums, drugposition, startdemog, demographics = surveyvars(year)

    # Convert fields in the demographic database to category variables.
    catvars = copy.deepcopy(demogs)
    catvars.remove('RESPID')

    # Create a new dataframe for translation of categorical values into
    # reader friendly values from the NSDUH dictionary.
    # Youth dictionary values are not considered.
    tmpdf = copy.deepcopy(df)
    pipeprep = Pipeline([('makecats', MakeCat(catvars))])
    tmpdf = pipeprep.transform(tmpdf)

    for fieldname in tmpdf.columns:
        newvallist = []
        if fieldname != 'RESPID' and fieldname != 'labels':
            for key, value in nsduhDecoder[fieldname]['values'].items():
                if value != "Youth":
                    newvallist.append(value)
            tmpdf[fieldname].cat.categories = newvallist

    # Generate crosstabs and Chi Square Tests and save to a text file
    f = open(filename, 'a')
    print('CROSSTABS AND CHISQUARE TEST RESULTS', file=f)
    for item in catvars:
        catcross = pd.crosstab(index=tmpdf['labels'], columns=tmpdf[item])
        print(catcross, '\n', file=f)
        catchi = stats.chi2_contingency(catcross)
        print(catchi, '\n\n', file=f)

    print('\n******\nEND OF SECTION\n******\n', file=f)

    # Generate crosstabs with percentages of columns
    print('COLUMN PERCENTAGE CROSSTABS', file=f)
    for item in catvars:
        catcross = pd.crosstab(index=tmpdf['labels'], columns=tmpdf[item], normalize='columns')
        print(catcross, '\n\n', file=f)

    print('\n******\nEND OF SECTION\n******\n', file=f)

    # Print out cluster pathways and save translation to a list
    print('CLUSTER PATHWAYS', file=f)
    centerpaths = []
    i = 0
    for i in range(0, len(model.cluster_centers_)):
        path = uncodepath(model.cluster_centers_[i].reshape(len(drugnums), len(drugnums)), drugnames)
        print('Path', i, ':', path, '\n', file=f)
        centerpaths.append(path)
    print('\n******\nEND OF SECTION\n******\n', file=f)
    f.close()

    return tmpdf


""" surveyvars()
surveyvars() returns drug name and drug number values associated with the NSDUH
survey for the input year.  The default year is 2016.
"""
### CHECK 2017 AND 2016 TO SEE IF VARIABLES ARE THE SAME!!!

def surveyvars(inyear='2016'):
    if inyear == '2016':
        # Set the relevant variables to read in
        ident = ['QUESTID2']
        
        # These are the AFU column names from the NSDUH survey
        rawafuvals = ['IRCIGAGE','IRCGRAGE','IRSMKLSSTRY','IRALCAGE','IRMJAGE','IRCOCAGE','IRCRKAGE','IRHERAGE',
                   'IRHALLUCAGE','IRINHALAGE','IRMETHAMAGE','IRPNRNMAGE','IRTRQNMAGE','IRSTMNMAGE','IRSEDNMAGE']
        
        # These are the AFU column names used in this study
        afuvals = ['IRTOBAGE','IRALCAGE','IRMJAGE','IRCOC2AGE','IRSCRIPAGE','IRHALLUCAGE','IRINHALAGE','IRHERAGE','IRMETHAMAGE']

        # This is the translation from NSDUH to drug names
        drugnames = {'IRTOBAGE': 'TOBACCO', 'IRALCAGE': 'ALCOHOL','IRMJAGE': 'MARIJUANA', 'IRCOC2AGE': 'COCAINE', 
                     'IRHERAGE': 'HEROIN', 'IRHALLUCAGE': 'HALLUCINOGEN', 'IRINHALAGE': 'INHALANTS', 
                     'IRMETHAMAGE': 'METHAMPHETAMINE', 'IRSCRIPAGE':'PRESCRIPTIONS', 'start':'start'}

        # This is the translation from NSDUH to position
        drugorder = {'start':0, 'IRTOBAGE':1, 'IRALCAGE':2, 'IRMJAGE':3, 'IRCOC2AGE':4, 'IRSCRIPAGE':5,
                     'IRHALLUCAGE':6, 'IRINHALAGE':7, 'IRHERAGE':8, 'IRMETHAMAGE':9}
        
        # This is the set of reference indices for the drugs for ordered arrays
        drugnums = {'start':0, 'TOBACCO':1, 'ALCOHOL':2, 'MARIJUANA':3, 'COCAINE':4, 'PRESCRIPTIONS':5,
                     'HALLUCINOGEN':6, 'INHALANTS':7, 'HEROIN':8, 'METHAMPHETAMINE':9}
        
        # This is the set of reference indices for the drugs for ordered arrays
        drugposition = {0:'start', 1:'TOBACCO', 2:'ALCOHOL', 3:'MARIJUANA', 4:'COCAINE', 5:'PRESCRIPTIONS',
                     6:'HALLUCINOGEN', 7:'INHALANTS', 8:'HEROIN', 9:'METHAMPHETAMINE'}
        
       # These are the column names from NSDUH
        startdemog = ['AGE2','SERVICE','CATAG6','IRSEX','IRMARIT','NEWRACE2','EDUHIGHCAT','IRWRKSTAT','GOVTPROG','INCOME','COUTYP4','AIIND102']
        demographics = ['AGE2','SVCFLAG','CATAG6','IRSEX','IRMARIT','NEWRACE2','EDUHIGHCAT','IRWRKSTAT','GOVTPROG','INCOME',
                        'COUTYP4','AIIND102','RESPID','ANALWT_C']

    else:
        # Set the relevant variables to read in
        ident = ['QUESTID2']
        
        # These are the AFU column names from the NSDUH survey
        rawafuvals = ['IRCIGAGE','IRCGRAGE','IRSMKLSSTRY','IRALCAGE','IRMJAGE','IRCOCAGE','IRCRKAGE','IRHERAGE',
                   'IRHALLUCAGE','IRINHALAGE','IRMETHAMAGE','IRPNRNMAGE','IRTRQNMAGE','IRSTMNMAGE','IRSEDNMAGE']
        
        # These are the AFU column names used in this study
        afuvals = ['IRTOBAGE','IRALCAGE','IRMJAGE','IRCOC2AGE','IRSCRIPAGE','IRHALLUCAGE','IRINHALAGE','IRHERAGE','IRMETHAMAGE']

        # This is the translation from NSDUH to drug names
        drugnames = {'IRTOBAGE': 'TOBACCO', 'IRALCAGE': 'ALCOHOL','IRMJAGE': 'MARIJUANA', 'IRCOC2AGE': 'COCAINE', 
                     'IRHERAGE': 'HEROIN', 'IRHALLUCAGE': 'HALLUCINOGEN', 'IRINHALAGE': 'INHALANTS', 
                     'IRMETHAMAGE': 'METHAMPHETAMINE', 'IRSCRIPAGE':'PRESCRIPTIONS'}

        # This is the translation from NSDUH to position
        drugorder = {'start':0, 'IRTOBAGE':1, 'IRALCAGE':2, 'IRMJAGE':3, 'IRCOC2AGE':4, 'IRSCRIPAGE':5,
                     'IRHALLUCAGE':6, 'IRINHALAGE':7, 'IRHERAGE':8, 'IRMETHAMAGE':9}
        
        # This is the set of reference indices for the drugs for ordered arrays
        drugnums = {'start':0, 'TOBACCO':1, 'ALCOHOL':2, 'MARIJUANA':3, 'COCAINE':4, 'PRESCRIPTIONS':5,
                     'HALLUCINOGEN':6, 'INHALANTS':7, 'HEROIN':8, 'METHAMPHETAMINE':9}
        
        # This is the set of reference indices for the drugs for ordered arrays
        drugposition = {0:'start', 1:'TOBACCO', 2:'ALCOHOL', 3:'MARIJUANA', 4:'COCAINE', 5:'PRESCRIPTIONS',
                     6:'HALLUCINOGEN', 7:'INHALANTS', 8:'HEROIN', 9:'METHAMPHETAMINE'}
        
       # These are the column names from NSDUH
        startdemog = ['AGE2','SERVICE','CATAG6','IRSEX','IRMARIT','NEWRACE2','EDUHIGHCAT','IRWRKSTAT','GOVTPROG','INCOME','COUTYP4','AIIND102','ANALWT_C']
        demographics = ['AGE2','SVCFLAG','CATAG6','IRSEX','IRMARIT','NEWRACE2','EDUHIGHCAT','IRWRKSTAT','GOVTPROG','INCOME',
                        'COUTYP4','AIIND102','RESPID','ANALWT_C']

    return ident, rawafuvals, afuvals, drugnames, drugorder, drugnums, drugposition, startdemog, demographics
    
    
""" valuedict()
valuedict() returns a dictionary of value maps for the demographic variable
"""

def valuedict(variable, value):
    dict = {
        'AGE2': 
            {1: '12', 2: '13', 3: '14', 4: '15', 5: '16', 6: '17', 7: '18', 8: '19', 9: '20',
             10: '21', 11: '22-23', 12: '24-25', 13: '26-29', 14: '30-34', 15: '35-49',
             16: '50-64', 17: '65+'},
        'SVCFLAG':
            {0: 'NO', 1: 'YES'},
        'CATAG6':
            {1: '12-17', 2: '18-25', 3: '26-34', 4: '35-49', 5: '50-64', 6: '65+'},
        'IRSEX':
            {1: 'MALE', 2: 'FEMALE'},
        'IRMARIT':
            {1: 'MARRIED', 2: 'WIDOWED', 3: 'DIVORCED OR SEPARATED', 4: 'NEVER MARRIED'},
        'NEWRACE2':
            {1: 'NONHISP WHITE', 2: 'NONHISP BLACK', 3: 'NONHISP NATIVE AM/AK', 4: 'NONHISP NATIVE HI/PAC ISL',
             5: 'NONHISP ASIAN', 6: 'NONHISP MULTIRACE', 7: 'HISPANIC'},
        'EDUHIGHCAT':
            {1: 'LESS HIGH SCHOOL', 2: 'HIGH SCHOOL GRAD', 3: 'SOME COLLEGE/ASSOC', 4: 'COLLEGE GRAD', 5: 'AGE 12-17'},
        'IRWRKSTAT':
            {1: 'FULL TIME', 2: 'PART TIME', 3: 'UNEMPLOYED', 4: 'OTHER', 99: 'AGE 12-14'},
        'GOVTPROG':
            {1: 'YES', 2: 'NO'},
        'INCOME':
            {1: '<$20K', 2: '$20K-$49,999', 3: '$50K-$74,999', 4: '$75K+'},
        'COUTYP4':
            {1: 'LARGE METRO', 2: 'SMALL METRO', 3: 'NONMETRO'},
        'AIIND102':
            {1: 'CENSUS BLOCK IN AMER INDIAN AREA', 2: 'CENSUS BLOCK NOT IN AMER INDIAN AREA'}
    }
    return dict[variable][value]


# The next set of functions helps visualization of results from classification studies
""" simpleprc()
Compute the predictions, prediction probabilities, and the accuracy scores
for the training and validation sets for the learned instance of the model.
This function produces the Precision-Recall curve.
"""
def simpleprc(model, Xtrain, ytrain, Xval, yval):
    # Compute model's predictions
    preds = model.predict(Xtrain)
    preds_val = model.predict(Xval)
    
    # Compute prediction probabilities
    proba = model.predict_proba(Xtrain)
    proba_val = model.predict_proba(Xval)

    # Compute the model's mean accuracy
    score = model.score(Xtrain, ytrain) 
    score_val = model.score(Xval, yval)

    # Calculate precision-recall statistics
    precision, recall, thresholds_prc = precision_recall_curve(yval, proba_val[:,1])
    auc_prc = auc(recall, precision)

    # Print values
    print('Training set mean accuracy:', score)
    print('Validation set mean accuracy:', score_val)
    print('Area under the Precision-Recall Curve:', auc_prc)
    
    return auc_prc

""" plotstats()
Display the confusion matrix, KS plot, ROC curve, and PR curve for the training 
and validation sets using metrics_plots.ks_roc_prc_plot

The red dashed line in the ROC and PR plots are indicative of the expected 
performance for a random classifier, which would predict postives at the 
rate of occurance within the data set
"""
def plotstats(model, Xtrain, ytrain, Xval, yval, targetnames):
    # Training confusion matrix and diagnostic curves
    print('Training Set Results')
    cmtx = confusion_matrix(ytrain, model.predict(Xtrain))
    metrics_plots.confusion_mtx_colormap(cmtx, targetnames, targetnames)
    rf_roc_prc_results = metrics_plots.ks_roc_prc_plot(ytrain, model.predict_proba(Xtrain)[:,1])
    
    # Validation confusion matrix and diagnostic curves
    print('Validation Set Results')
    cmtx = confusion_matrix(yval, model.predict(Xval))
    metrics_plots.confusion_mtx_colormap(cmtx, targetnames, targetnames)
    rf_roc_prc_results = metrics_plots.ks_roc_prc_plot(yval, model.predict_proba(Xval)[:,1])
    
    return
    

"""
plot_confusion_matrix()
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
"""
def plot_confusion_matrix(cm, classes, ax,
                      normalize=False,
                      title='Confusion matrix',
                      cmap=plt.cm.Blues):
    print(cm)
    print('')

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.sca(ax)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')