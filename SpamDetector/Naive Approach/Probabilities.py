# %%
# Imports
import numpy as np
import pandas as pd
from time import time

# %%
# Constants
TRAIN_MAT_PATH = 'SpamData/02_Training/Train-MATRIX.txt'
TEST_MAT_PATH = 'SpamData/02_Training/Test-MATRIX.txt'
TRAIN_PROB_SPAM = 'SpamData/03_Testing/Train-SPAMPROB.txt'
TRAIN_PROB_HAM = 'SpamData/03_Testing/Train-HAMPROB.txt'
TRAIN_PROB_ALL = 'SpamData/03_Testing/Train-ALLPROB.txt'
TEST_FEATURES = 'SpamData/03_Testing/TEST-FEATURES.txt'
TEST_TARGET = 'SpamData/03_Testing/TEST-TARGET.txt'
SIZE = 2500

# %%
# Loading the matrices
sparse_train_data = np.loadtxt(TRAIN_MAT_PATH, delimiter=' ', dtype=int)
sparse_test_data = np.loadtxt(TEST_MAT_PATH, delimiter=' ', dtype=int)
print(sparse_test_data[:5])
print(sparse_train_data[:5])

# %%
# %%
start = time()
# Bulding a full matrix as a dataFrame


def buildFullMatrix(sparse_matrix, size, doc_idx=0, word_idx=1, category_idx=2, freq_idx=3):
    # Buidling rows and  columns
    column_names = ['DOC_ID', 'CATEGORY']+list(range(size))
    row_names = np.unique(sparse_matrix[:, 0])
    full_matrix = pd.DataFrame(columns=column_names, index=row_names)
    full_matrix.fillna(value=0, inplace=True)
    for i in range(sparse_matrix.shape[0]):
        doc_id = sparse_matrix[i][doc_idx]
        word_id = sparse_matrix[i][word_idx]
        category = sparse_matrix[i][category_idx]
        frequency = sparse_matrix[i][freq_idx]
        full_matrix.at[doc_id, 'DOC_ID'] = doc_id
        full_matrix.at[doc_id, word_id] = frequency
        full_matrix.at[doc_id, 'CATEGORY'] = category
    full_matrix.set_index('DOC_ID', inplace=True)
    return full_matrix


full_train_data = buildFullMatrix(sparse_train_data, SIZE)
print(full_train_data.head())
print(time()-start)

# %%
# Calculating Probabilities
# P(S|W)=P(W|S)*P(S)/P(W)
'''
P(S|W) -- Probability of message being spam given that it has the word
P(W|S) -- Probability of message having the word given that it is spam
P(S) -- Probability of message being spam
P(W) -- Probability of word occuring in all mails
'''

# Probability of Spam Message P(S)
prob_spam = full_train_data.CATEGORY.sum()/full_train_data.CATEGORY.size
print(prob_spam)


# Calculating total number of words
# Removing the spam boolean and calculating the sum of frequencies
features = full_train_data.loc[:, full_train_data.columns != 'CATEGORY']
# Count of total words
email_lengths = features.sum(axis=1)
word_count = email_lengths.sum()
# Count of words in spam/ham mails
spam_lengths = email_lengths[full_train_data.CATEGORY == 1]
ham_lengths = email_lengths[full_train_data.CATEGORY == 0]
spam_wc = spam_lengths.sum()
ham_wc = ham_lengths.sum()
print(spam_wc, ham_wc)

# %%
# Calculating word frequencies
spam_train_set = full_train_data.loc[full_train_data.CATEGORY == 1]
# 1 is added for laplace smoothing
total_spam_tokens = spam_train_set.sum(axis=0)+1
print(total_spam_tokens.tail())
ham_train_set = full_train_data.loc[full_train_data.CATEGORY == 0]
total_ham_tokens = ham_train_set.sum(axis=0)+1
print(total_ham_tokens.tail())

# %%
# Calculating P(W|S) P(W|S') P(W)
prob_word_spam = total_spam_tokens/(spam_wc+SIZE)  # P(W|S)
prob_word_ham = total_ham_tokens/(ham_wc+SIZE)  # P(W|S')
print(prob_word_ham, prob_word_spam)
prob_word = features.sum(axis=0)/word_count  # P(W)
np.savetxt(TRAIN_PROB_SPAM, prob_word_spam)
np.savetxt(TRAIN_PROB_HAM, prob_word_ham)
np.savetxt(TRAIN_PROB_ALL, prob_word)

# %%
# Building the final test dataset
full_test_data = buildFullMatrix(sparse_test_data, SIZE)
print(full_test_data.shape)
print(full_test_data.head())
features = full_test_data.loc[:, full_test_data.columns != 'CATEGORY']
target = full_test_data.CATEGORY
print(features.head())
np.savetxt(TEST_FEATURES, features)
np.savetxt(TEST_TARGET, target)
