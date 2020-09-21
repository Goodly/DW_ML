
# coding: utf-8

# ##### Last updated: 2019-03-02

# ### Upload DF corpus

# ##### Way #1 (not preferred)
# 
# Manually upload a zip file of 2017 documents and then unzip it. 
# 
# It takes a couple of minutes to upload the whole thing.
# 
# Notes:
# - A residue folder `__MACOSX` is created when unzipping; not sure why...
# - The following error is encountered when unzipping (maybe related to above?):
# 
# ```
# IOPub data rate exceeded.
# The notebook server will temporarily stop sending output
# to the client in order to avoid crashing it.
# To change this limit, set the config variable
# `--NotebookApp.iopub_data_rate_limit`.
# 
# Current values:
# NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
# NotebookApp.rate_limit_window=3.0 (secs)
# ```

# In[ ]:


#!unzip DocumentsParsed-2017.zip


# ##### Way #2 (better)
# 
# Create a new directory `df-corpus` and copy the whole corpus from `S3` into this directory.

# In[ ]:


#!mkdir df-corpus


# In[ ]:


#!aws s3 cp s3://tagworks.thusly.co/decidingforce/corpus/ ./df-corpus --recursive


# In[ ]:


#!find df-corpus/* -maxdepth 0 -type d | wc -l # See how many folders are under df-corpus


# ### Install Stanford CoreNLP

# In[ ]:


#!wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip


# In[ ]:


#!unzip stanford-corenlp-full-2018-10-05.zip


# ### Install Java

# In[ ]:


#!java -version


# ### Upload prop file

# `df-classifier.prop` tells the CRF classifier "how" to go about classifying. NER Feature Factory lists all the possible parameters that can be tuned: https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ie/NERFeatureFactory.html
# 
# Right now I'm manually uploading it here.

# ### Download stopwords and wordnet

# It takes a few seconds to load...

# In[1]:


import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# ### Create lemmatizer

# In[2]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# ### Import libraries

# In[3]:


import gzip, json, nltk, os, re, string
from nltk.corpus import stopwords
import pandas as pd
import time


# ### Define auxiliary functions

# In[4]:


def store_annotations(path_to_data):
    with gzip.open(os.path.join(path_to_data, "annotations.json.gz"), 
                   mode='rt', 
                   encoding='utf8') as unzipped: 
        annotations = json.load(unzipped)
    return(annotations)

def store_text(path_to_data):
    with gzip.open(os.path.join(path_to_data, "text.txt.gz"), 
                   mode='rt', 
                   encoding='utf8') as unzipped: 
        text = unzipped.read()
    return(text)

def gen_lst_tags(annotations):
    lst_tagged_text = []
    for e1 in annotations["tuas"]:
        for e2 in annotations["tuas"][e1]:
            for e3 in annotations["tuas"][e1][e2]:
                lst_tagged_text += [[e1, e3[0], e3[1], e3[2]]]
    lst_tagged_text = sorted(lst_tagged_text, key = lambda x: x[1])
    return(lst_tagged_text)

def reorganize_tag_positions(tag_positions):
    keep_going = 1
    while keep_going:
        keep_going = 0
        p = 0
        tag_positions_better = []
        while p < len(tag_positions) - 1:
            if tag_positions[p][1] < tag_positions[p+1][0] - 1:
                tag_positions_better += [tag_positions[p]]
                p += 1
                if p == len(tag_positions) - 1:
                    tag_positions_better += [tag_positions[p]]
            elif tag_positions[p][1] >= tag_positions[p+1][1]:
                tag_positions_better += [tag_positions[p]]
                p += 2
                keep_going = 1
                if p == len(tag_positions) - 1:
                    tag_positions_better += [tag_positions[p]]
            else:
                tag_positions_better += [[tag_positions[p][0], tag_positions[p+1][1]]]
                p += 2
                keep_going = 1
                if p == len(tag_positions) - 1:
                    tag_positions_better += [tag_positions[p]]
        tag_positions = tag_positions_better.copy()
    return(tag_positions_better)

def gen_lst_untagged(tag_positions_better, text):
    lst_untagged_text = []
    p0 = 0
    for p in tag_positions_better:
        #lst_untagged_text += [['Untagged', p0, p[0]-1, text[p0:p[0]]]]
        lst_untagged_text += [['O', p0, p[0]-1, text[p0:p[0]]]]
        p0 = p[1] + 1
    lst_untagged_text = [e for e in lst_untagged_text]
    return(lst_untagged_text)


# ### Define main functions

# ##### `gen_word_tag_lst`
# 
# This function allows users to specify whether to:
# 
# - remove stopwords or not
# - use POS tags or not
# - focus on one label and treat everything else as other (e.g., Protester vs. O) or not
#     - in other words, binary classification vs. multiclass classification
# 
# It is possible to add more flexibility to this function to also allow users to specificy whether to:
# 
# - remove punctuation or not (removing punctuation is default right now)
# - transform words to lowercase or not (transforming to lowercase is default right now)
# - lemmatize words or not (lemmatizing is default right now)
# 
# ##### `write_to_tsv`
# 
# This function allows users to specify which set of documents to use for the train and test datasets. The `tsv` file generated at the end includes words from documents between `start_index` and `end_index`. `end_index` can be as high as the number of documents in the corpus (here, 8094). The function needs to be run twice, once for generating the train dataset and once for generating the test dataset. 

# In[5]:


def gen_word_tag_lst(path_to_data, remove_stop_words, use_pos, focus, focus_word):
    
    # Store annotations
    annotations = store_annotations(path_to_data)

    # Store full text
    text = store_text(path_to_data)
    
    # Generate list of tagged text
    lst_tagged_text = gen_lst_tags(annotations)
    
    # Generate list of tag positions
    tag_positions = sorted([e[1:3] for e in lst_tagged_text])
    
    # Reorganize tag positions
    tag_positions_better = reorganize_tag_positions(tag_positions)
        
    # Generate list of untagged text
    lst_untagged_text = gen_lst_untagged(tag_positions_better, text)
    
    # Generate list of tagged and untagged text
    lst_full_text = sorted(lst_tagged_text + lst_untagged_text, 
                           key = lambda x: x[1])
    
    # Add part-of-speech (POS) tags
    for i, e in enumerate(lst_full_text):
        tokens = nltk.word_tokenize(e[3])
        pos_document = nltk.pos_tag(tokens)
        lst_full_text[i][3] = pos_document
    
    # Generate table that stores info on what is going to be excluded from strings
    table = str.maketrans({key: " " for key in set(string.punctuation + 
                                                   "\n" + "\xa0" + 
                                                   "“" + "’" + "–" + 
                                                   "\u201d" + "\u2018" + "\u2013" + "\u2014")})
    
    # Store English stop words
    stopwords_en = stopwords.words('english')
    
    # Generate final list to be converted to tsv format (lemmatize on the way)
    lst = []
    for e in lst_full_text:
        for token in e[3]:
            # Remove punctuation, transform to lower case, and strip any white space at start/end
            token = (token[0].translate(table).lower().strip(), token[1])
            if token[0]:
                if remove_stop_words:
                    if token[0] not in stopwords_en:
                        if focus:
                            if e[0] == focus_word:
                                if use_pos:
                                    lst += [lemmatizer.lemmatize(token[0]) + "\t" + token[1] + "\t" + e[0]]
                                else:
                                    lst += [lemmatizer.lemmatize(token[0]) + "\t" + e[0]]
                            else:
                                if use_pos:
                                    lst += [lemmatizer.lemmatize(token[0]) + "\t" + token[1] + "\t" + 'O']
                                else:
                                    lst += [lemmatizer.lemmatize(token[0]) + "\t" + 'O']
                        else:
                            if use_pos:
                                lst += [lemmatizer.lemmatize(token[0]) + "\t" + token[1] + "\t" + e[0]]
                            else:
                                lst += [lemmatizer.lemmatize(token[0]) + "\t" + e[0]]
                else:
                    if focus:
                        if e[0] == focus_word:
                            if use_pos:
                                lst += [lemmatizer.lemmatize(token[0]) + "\t" + token[1] + "\t" + e[0]]
                            else:
                                lst += [lemmatizer.lemmatize(token[0]) + "\t" + e[0]]
                        else:
                            if use_pos:
                                lst += [lemmatizer.lemmatize(token[0]) + "\t" + token[1] + "\t" + 'O']
                            else:
                                lst += [lemmatizer.lemmatize(token[0]) + "\t" + 'O']
                    else:
                        if use_pos:
                            lst += [lemmatizer.lemmatize(token[0]) + "\t" + token[1] + "\t" + e[0]]
                        else:
                            lst += [lemmatizer.lemmatize(token[0]) + "\t" + e[0]]
    return(lst)

def write_to_tsv(path_to_tsv, 
                 path_to_data, 
                 train_or_test, 
                 start_index,
                 end_index,
                 remove_stop_words = True, 
                 use_pos = True,
                 focus = True, 
                 focus_word = "Protester"):
    p = 0
    with open(os.path.join(path_to_tsv, train_or_test), 'w') as file:        
        for root, dirs, files in os.walk(path_to_data):
            if not dirs and "text.txt.gz" in files and "annotations.json.gz" in files:
                if start_index <= p and end_index > p:
                    word_tag_lst = gen_word_tag_lst(root, remove_stop_words, use_pos, focus, focus_word)
                    # Filter out Useless and ToBe tags
                    word_tag_lst = list(filter(lambda x: 'Useless' not in x and 'ToBe' not in x, word_tag_lst))
                    for e in word_tag_lst:
                        file.write(e + '\n')
                    if word_tag_lst:
                        file.write('\n')
                p += 1


# ### Generate train and test data

# In[6]:


#path_to_data_2017 = "./DocumentsParsed-2017"
path_to_data = "./df-corpus"

path_to_tsv = "."


# In[7]:


# Generate train data
write_to_tsv(path_to_tsv, path_to_data, train_or_test = 'train.tsv', 
             start_index = 0, end_index = 500,
             remove_stop_words = True, use_pos = True, focus = False, focus_word = "Protester")

# Generate test data
write_to_tsv(path_to_tsv, path_to_data, train_or_test = 'test.tsv', 
             start_index = 500, end_index = 600,
             remove_stop_words = True, use_pos = True, focus = False, focus_word = "Protester")


# ### Train and test model

# In[8]:


start_time = time.time()

# Train model

get_ipython().system('java -Xmx16g -cp "./stanford-corenlp-full-2018-10-05/*" edu.stanford.nlp.ie.crf.CRFClassifier -prop ./df-classifier.prop')

print((time.time()-start_time)/60)


# In[9]:


# Test model

get_ipython().system('java -Xmx16g -cp "./stanford-corenlp-full-2018-10-05/*" edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ./custom-tagger.ser.gz -testFile ./test.tsv -outputFormat tsv 1> "./test-results/0-500-500-600-RP-ULC-L-RSW-UPOS-DNF-NA.tsv"')

# num1: train start
# num2: train end
# num3: test start
# num4: test end

# RP: remove punctuation
# DNRP: do not remove punctuation

# ULC: use lower case
# DNULC: do not use lower case

# L: lemmatize
# DNL: do not lemmatize

# RSW: remove stop words
# DNRSW: do not remove stop words

# UPOS: use part-of-speech
# DNUPOS: do not use part-of-speech

# F: focus
# DNF: do not focus

# Pr: Protester
# O: Opinioner
# C: Camp
# S: Strategy
# I: Info
# G: Government
# P: Police
# L: Legal_Action
# NA: not applicable


# ### Check model performance

# In[10]:


df = pd.read_csv("./test-results/0-500-500-600-RP-ULC-L-RSW-UPOS-DNF-NA.tsv", 
                 sep = '\t',
                 names = ["word", "obs", "pred"])

#                      TP, FP, TN, FN
d = {"O"            : [0,  0,  0,  0],
     "Protester"    : [0,  0,  0,  0],
     "Opinionor"    : [0,  0,  0,  0],
     "Camp"         : [0,  0,  0,  0],
     "Strategy"     : [0,  0,  0,  0],
     "Info"         : [0,  0,  0,  0],
     "Government"   : [0,  0,  0,  0],
     "Police"       : [0,  0,  0,  0],
     "Legal_Action" : [0,  0,  0,  0]}

for index, row in df.iterrows():
    if row['obs'] == row['pred']:
        d[row['pred']][0] += 1
        for key in d.keys():
            if key != row['pred']:
                d[key][2] += 1
    if row['obs'] != row['pred']:
        d[row['pred']][1] += 1
        d[row['obs']][3] += 1

for key in d.keys():
    if d[key][0] == 0 and d[key][1] == 0 and d[key][3] == 0:
        continue
    else:
        try:
            accuracy     = (d[key][0] + d[key][2])/sum(d[key])
        except:
            accuracy     = 0
        try:
            precision    = d[key][0]/(d[key][0] + d[key][1])
        except:
            precision    = 0
        try:
            recall       = d[key][0]/(d[key][0] + d[key][3])
        except: 
            recal        = 0
        try:
            specificity  = d[key][2]/(d[key][1] + d[key][2])
        except:
            specificity  = 0
        try:
            f1_score     = 2*precision*recall/(precision+recall)
        except:
            f1_score     = 0

        print("TP, FP, TN, FN for " + key + " are: " + str(d[key]))
        print("Accuracy for "       + key + " is: "  + str(accuracy))
        print("Precision for "      + key + " is: "  + str(precision))
        print("Recall for "         + key + " is: "  + str(recall))
        print("Specificity for "    + key + " is: "  + str(specificity))
        print("F1 score for "       + key + " is: "  + str(f1_score) + "\n")


# ### Sandbox

# In[ ]:


if False:
    # Count number of articles in the corpus
    count = 0
    for root, dirs, files in os.walk("./df-corpus"):
        if not dirs and "text.txt.gz" in files and "annotations.json.gz" in files:
            count += 1
    print(count)


# In[ ]:


if False:
    # The slicing in the first for loop can be used 
    #   to select only those directories from a specific city (e.g., 0:11 is Albany)
    # The slicing in the second for loop can be used 
    #   to select the number of articles from that specific city.
    #   This is relevant when splitting articles from a specific city
    #   into train and test batches. 
    def write_to_tsv_alt(path_to_tsv, 
                         path_to_data, 
                         train_or_test, 
                         start1 = 0,
                         end1 = 1342,
                         start2 = 0,
                         end2 = None,
                         remove_stop_words = True, 
                         focus = True, 
                         focus_word = "Protester", 
                         use_pos = True):
        with open(os.path.join(path_to_tsv, train_or_test), 'w') as file:
            for f in sorted(os.listdir(path_to_data))[start1:end1]:
                if f != ".DS_Store":
                    for sf in sorted(os.listdir(os.path.join(path_to_data, f)))[start2:end2]:
                        if sf != ".DS_Store":
                            path = os.path.join(path_to_data, f, sf)
                            word_tag_lst = gen_word_tag_lst(path, remove_stop_words, focus, focus_word, use_pos)
                            # Filter out Useless and ToBe tags
                            word_tag_lst = list(
                                filter(lambda x: 'Useless' not in x and 'ToBe' not in x, word_tag_lst))
                            for e in word_tag_lst:
                                file.write(e + '\n')
                            if word_tag_lst:
                                file.write('\n')


# In[ ]:


if False:
    # The slicing in the first for loop can be used 
    #   to select only those directories from a specific city (e.g., 0:11 is Albany)
    # The slicing in the second for loop can be used 
    #   to select the number of articles from that specific city.
    #   This is relevant when splitting articles from a specific city
    #   into train and test batches. 
    count = 0
    for f in sorted(os.listdir(path_to_data))[0:11]:
        if f != ".DS_Store":
            for sf in sorted(os.listdir(os.path.join(path_to_data, f)))[0:]:
                if sf != ".DS_Store":
                    path = os.path.join(path_to_data, f, sf)
                    print(path)
                    count += 1
    print(count)

