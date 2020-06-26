###
# Naive Bayes classifier for text documents
###

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import math
import os
from collections import Counter
import datetime
import zipfile

start_time = datetime.datetime.now()
print(start_time)

# Step 1: Pre processing
#   a.  Read first 500 files from every folder as a train_class
#   b.  Form a vocabulary and calculate length by merging all class docs (this is for laplace correction)
#   c.  Form a test_class by adding remaining 500 docs from each folder

# a.
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stop_words = set(stopwords.words('english'))

path = os.path.join(os.getcwd(),'20_newsgroups')
if not os.path.exists(path):
    with zipfile.ZipFile('20_newsgroups.zip') as myzip:
        myzip.extractall()

class_name_list = []
for filename in os.listdir(path):
    class_name_list.append(filename)

train_class_tok_dict = {}
train_class_tok_len_dict = {}
train_class_doc_dict = {}

for class_name in class_name_list:
    class_path = os.path.join(path, class_name)
    print(class_path)
    doc_list = []
    train_class_doc_list = []
    doc_count = 0
    for filename in os.listdir(class_path):
        f = open(os.path.join(class_path, filename), "r")
        train_class_doc_list.append(filename)
        doc = f.read()
        f.close()
        doc = doc.lower()
        doc_list.append(doc)
        doc_count = doc_count + 1
        if doc_count == 500:
            break

    train_class_doc_dict[class_name] = train_class_doc_list
    tokens_list = []
    for doc in doc_list:
        tokens = tokenizer.tokenize(doc)
        tokens = [x for x in tokens if x not in stop_words]
        tokens_list.extend(tokens)

    tokens_dict = dict(Counter(tokens_list))
    train_class_tok_dict[class_name] = tokens_dict

vocab_list = []
for class_name,tokens_dict in train_class_tok_dict.items():
    tokens = list(tokens_dict.keys())
    tokens_len = sum(tokens_dict.values())
    train_class_tok_len_dict[class_name] = tokens_len
    vocab_list.extend(tokens)

# b. Vocabulary and length
vocab_set = set(vocab_list)
vocab_len = (len(vocab_set))
print("vocab_len",vocab_len)
print("class_tok_len_dict", train_class_tok_len_dict)

#c. Form test documents list
# Here we will deal with each folder separately to calculate accuracy
# Form a list of remaining docs of folder; calculate results and find correctly classification # for this folder
# Repeat same for all folders and take combine accuracy.

test_class_doc_dict = {}
correct_class_dict = {}
for class_name in class_name_list:
    class_path = os.path.join(path,class_name)
    print(class_path)
    train_class_doc_list = train_class_doc_dict[class_name]
    doc_list = []
    test_class_doc_list = []
    correct_class = 0

    for filename in os.listdir(class_path):
        if filename not in train_class_doc_list:
            f = open(os.path.join(class_path, filename), "r")
            test_class_doc_list.append(filename)
            doc = f.read()
            f.close()
            doc = doc.lower()
            doc_list.append(doc)

    test_class_doc_dict[class_name] = test_class_doc_list
    tokens_list = []
    for doc in doc_list:
        tokens = tokenizer.tokenize(doc)
        tokens = [x for x in tokens if x not in stop_words]
        tokens_list.extend(tokens)
        tokens_dict = dict(Counter(tokens_list))
        doc_prob_dict = {}
        for cls_nm in class_name_list:
            doc_prob = 1
            doc_log_prob = 0
            tok_count_tot = train_class_tok_len_dict[cls_nm]
            # adding laplace correction
            tok_count_tot = tok_count_tot + vocab_len
            for tok, tok_freq in tokens_dict.items():
                tok_count_dict = train_class_tok_dict[cls_nm]
                try:
                    tok_count = tok_count_dict[tok]
                except KeyError:
                    tok_count = 0

                # adding laplace correction
                tok_count = tok_count + 1
                tok_prob = tok_count / tok_count_tot

                #make power of # times it occurs in current doc
                #word_freq = tok_freq
                #word_prob = pow(word_prob,word_occur)
                tok_log_prob = tok_freq * math.log10(tok_prob)

                #docprob = docprob * word_prob
                doc_log_prob = doc_log_prob + tok_log_prob

            #docprob_dict[k] = docprob
            doc_prob_dict[cls_nm] = doc_log_prob

        # find max prob and decide class
        class_res = max(doc_prob_dict, key=lambda p: doc_prob_dict[p])

        if class_res == class_name:
            correct_class = correct_class + 1

    correct_class_dict[class_name] = correct_class
    print(class_name, correct_class)
print("correct_class_dict",correct_class_dict)

correct_classified_docs = sum(list(correct_class_dict.values()))
print("correct_classified_docs", correct_classified_docs)
tot_test_docs = 0
for class_name in class_name_list:
    test_doc_list = test_class_doc_dict[class_name]
    tot_test_docs = tot_test_docs + len(test_doc_list)
print("total test docs", tot_test_docs)
model_accuracy = (correct_classified_docs / tot_test_docs) * 100
print("acc", model_accuracy)

end_time = datetime.datetime.now()
print("time elapsed", end_time - start_time)