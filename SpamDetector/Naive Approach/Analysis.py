# %%
# Import Statements
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from PIL import Image
from wordcloud import WordCloud


# %%
# Constants
HAM_1_PATH = 'SpamData/01_Processing/spam_assassin_corpus/easy_ham_1'
HAM_2_PATH = 'SpamData/01_Processing/spam_assassin_corpus/easy_ham_2'
SPAM_1_PATH = 'SpamData/01_Processing/spam_assassin_corpus/spam_1'
SPAM_2_PATH = 'SpamData/01_Processing/spam_assassin_corpus/spam_2'
WHALE_IMG = 'SpamData/01_Processing/wordcloud_resources/whale-icon.png'
SKULL_IMG = 'SpamData/01_Processing/wordcloud_resources/skull-icon.png'
THUMBSUP_IMG = 'SpamData/01_Processing/wordcloud_resources/thumbs-up.png'
THUMBSDOWN_IMG = 'SpamData/01_Processing/wordcloud_resources/thumbs-down.png'
JSON_PATH = 'SpamData/02_Training/EMAIL-DATA.json'
CSV_PATH = 'SpamData/02_Training/VOCABULARY.csv'
TRAIN_MAT_PATH = 'SpamData/02_Training/Train-MATRIX.txt'
TEST_MAT_PATH = 'SpamData/02_Training/Test-MATRIX.txt'
SIZE = 2500


# MAIL CLASSIFIERS
SPAM_CAT = 1
HAM_CAT = 0


# %%
def emailBodyGenerator(path):
    # Walk returns a tuple with  following data
    for root, dirnames, filenames in os.walk(path):
        # for each file we obtain its path first
        for file_name in filenames:
            filepath = os.path.join(root, file_name)
            # fetch only the body data
            stream = open(filepath, encoding='latin-1')
            is_body = False
            data = []
            for line in stream:
                if is_body:
                    data.append(line)
                elif(line == "\n"):
                    is_body = True
            stream.close()
            email_body = "\n".join(data)
            yield file_name, email_body

# %%


def emaildataFrameBuilder(path, classification):
    # rows consists of data
    rows = []
    # row names consist of file names
    row_names = []
    for file_name, email_body in emailBodyGenerator(path):
        rows.append({'MESSAGE': email_body, 'CATEGORY': classification})
        row_names.append(file_name)
    # building the dataframe
    return pd.DataFrame(rows, index=row_names)


# %%
# Collect all Mails
ham_emails = emaildataFrameBuilder(HAM_1_PATH, HAM_CAT)
ham_emails = ham_emails.append(emaildataFrameBuilder(HAM_2_PATH, HAM_CAT))
spam_emails = emaildataFrameBuilder(SPAM_1_PATH, SPAM_CAT)
spam_emails = spam_emails.append(emaildataFrameBuilder(SPAM_2_PATH, SPAM_CAT))
data = pd.concat([spam_emails, ham_emails])
data.head


# %%
# Check for null values
data['MESSAGE'].isnull().values.any()
# Check for empty emails and retreiving their indices
data[data['MESSAGE'].str.len() == 0].index
# Remove empty entries
# Dropped the values
data.drop(['cmds'], inplace=True)
data.shape

# %%
# Assigning numerical IDS to all emails
docuemt_ids = range(0, len(data.index))
data['DOC_ID'] = docuemt_ids
# data['DOC_ID']
data['FILE_NAME'] = data.index
data.set_index('DOC_ID', inplace=True)
data.head
# Save Data to json file
# data.to_json(JSON_PATH)

# %%
# Visualizing Data - PieChart
spam_count = data.CATEGORY.value_counts()[1]
ham_count = data.CATEGORY.value_counts()[0]
custom_colors = ['#ff7675', '#74b9ff']
category_names = ['SPAM', 'LEGIT MAIL']
sizes = [spam_count, ham_count]
plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names, textprops={
        'fontsize': 6}, autopct='%1.1f%%', explode=[0, 0.1], startangle=90, colors=custom_colors)

# %%
# Visualizing Data DonutChart
spam_count = data.CATEGORY.value_counts()[1]
ham_count = data.CATEGORY.value_counts()[0]
custom_colors = ['#ff7675', '#74b9ff']
category_names = ['SPAM', 'LEGIT MAIL']
sizes = [spam_count, ham_count]
plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names, textprops={
        'fontsize': 6}, autopct='%1.0f%%', pctdistance=0.8, startangle=90, colors=custom_colors)
circle = plt.Circle((0, 0), radius=0.6, fc='white')
plt.gca().add_artist(circle)

# %%
# Natural Language Processing
'''
1.Convert Everything to LowerCase
2.Tokenizing
3.Removing stop words like 'the'
4.Strippin redundant data like HTML tags
5.Stemming the words (go,goes,going has root word 'go')
6.Removing Punctuation
'''
# nltk.download('punkt')
# nltk.download('stopwords')
# Stemmer
stemmer = PorterStemmer()
# Test String
msg = "All work and no play  makes jack a dull boy! To be or not to be"
# Lowercasing the sentence
words = word_tokenize(msg.lower())
# output list
filtered_words = []
# Retreiving list of stop words
stop_words = set(stopwords.words('english'))
# Using the features for each word in loop
for word in words:
    if word not in stop_words and word.isalpha():
        filtered_words.append(stemmer.stem(word))
# Final Output
print(filtered_words)

# %%
# Removing HTML tags from text
soup = BeautifulSoup(data.at[2, 'MESSAGE'], 'html.parser')
print(soup.get_text())

# %%


def filterMessage(message):
    # Filtering the message based on previous methodologies
    soup = BeautifulSoup(message, 'html.parser')
    message = soup.get_text()
    words = word_tokenize(message.lower())
    filtered_words = []
    for word in words:
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))
    return filtered_words


# %%
# Bulding the lists
nested_list = data.MESSAGE.apply(filterMessage)
spam_ids = data[data.CATEGORY == 1].index
ham_ids = data[data.CATEGORY == 0].index
spam_nested_list = nested_list.loc[spam_ids]
ham_nested_list = nested_list.loc[ham_ids]
spam_nested_list.shape

# %%
# EXtracting the words
spam_list = [j for i in spam_nested_list for j in i]
spam_words = pd.Series(spam_list).value_counts()
ham_list = [j for i in ham_nested_list for j in i]
ham_words = pd.Series(ham_list).value_counts()
print(spam_words)
print(ham_words)


# %%
# Building a Sample WordCloud
# nltk.download('gutenberg')
# nltk.download('shakespeare')
example_corpus = nltk.corpus.gutenberg.words('melville-moby_dick.txt')
# setting all words in a list
sample_list = [''.join(word for word in example_corpus)]
# joining them as a string with spaces
sample_novel_string = ' '.join(sample_list)
# Building wordcloud using image
icon = Image.open(WHALE_IMG)
image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
image_mask.paste(icon, box=icon)
rgb_array = np.array(image_mask)
# generating sample word cloud
plt.figure(figsize=(16, 8))
word_cloud = WordCloud(mask=rgb_array, background_color='white',
                       max_words=400, colormap='ocean').generate(sample_novel_string)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# %%
start = time.time()
# WordCloud Sample #2
# Retrieving sample string
example_corpus = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
sample_list = [''.join(example_corpus)]
sample_novel_string = ' '.join(sample_list)
# Se3tting up the image for wordcloud
icon = Image.open(SKULL_IMG)
image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
image_mask.paste(icon, box=icon)
rgb_array = np.array(image_mask)
# generating wordcloud
plt.figure(figsize=(16, 8))
word_cloud = WordCloud(mask=rgb_array, background_color='black',
                       colormap='autumn', max_words=500).generate(sample_novel_string)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()
print(time.time()-start)


# %%
# Bulding Wordclouds of the official samples
ham_string = ' '.join(ham_list)
# Image building
icon = Image.open(THUMBSUP_IMG)
image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
image_mask.paste(icon, box=icon)
rgb_array = np.array(image_mask)
# plotting the wordcloud
plt.figure(figsize=(16, 8))
word_cloud = WordCloud(mask=rgb_array, background_color='white',
                       colormap='winter', max_words=500).generate(ham_string)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


spam_string = ' '.join(spam_list)
# Image building
icon = Image.open(THUMBSDOWN_IMG)
image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
image_mask.paste(icon, box=icon)
rgb_array = np.array(image_mask)
# plotting the wordcloud
plt.figure(figsize=(16, 8))
word_cloud = WordCloud(mask=rgb_array, background_color='white',
                       colormap='gist_heat', max_words=1500).generate(spam_string)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %%
# Getitng all the words in a list
word_list = data.MESSAGE.apply(filterMessage)
all_words = [j for i in word_list for j in i]
unique_words = pd.Series(all_words).value_counts()
print(unique_words.head())

# %%
# Creating a subset from the list
ids = list(range(0, SIZE))
frequent_words = unique_words[:SIZE]
vocab = pd.DataFrame({'WORD': frequent_words.index.values}, index=ids)
vocab.index.name = 'ID'
# Save as CSV File
#vocab.to_csv(CSV_PATH, header=vocab.WORD.name,index_label=vocab.index.name)

# %%
# Creating a DataFrame and training/testing datasets
words_df = pd.DataFrame.from_records(word_list.tolist())
xtrain, xtest, ytrain, ytest = train_test_split(
    words_df, data.CATEGORY, test_size=0.3, random_state=42)
xtrain.index.name = 'ID'
xtest.index.name = 'ID'
xtrain.head()
# Building an indexed list to access the words
words_index = pd.Index(vocab.WORD)
print(words_index)
print(xtest.shape)

# %%


def buildSparseMatrix(df, indexed_words, labels):
    '''
    Returns a Sparse Matrix

    Parameters:

    df -- DataFrame consisting of stemmed words ordered by id

    indexed_words -- index of words ordered by ids

    labels -- Specify if email is spam or not(1 for spam and 0 for not spam) as a Series

    '''
    row_count = df.shape[0]
    col_count = df.shape[1]
    word_set = set(indexed_words)
    dict_list = []
    for i in range(row_count):
        for j in range(col_count):
            word = df.iat[i, j]
            if word in word_set:
                doc_id = df.index[i]
                word_id = indexed_words.get_loc(word)
                category = labels.at[doc_id]
                item = {'LABEL': category, 'DOC_ID': doc_id,
                        'OCCURENCE': 1, 'WORD_ID': word_id}
                dict_list.append(item)
    return pd.DataFrame(dict_list)


words_df = buildSparseMatrix(xtrain, words_index, ytrain)
print(words_df)


# %%
# Building the final sparse Matrix for train dataset
start = time.time()
sparse_matrix_train = words_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum()
sparse_matrix_train = sparse_matrix_train.reset_index()
print(sparse_matrix_train.head())
print(sparse_matrix_train.shape)
print(time.time()-start)
np.savetxt(TRAIN_MAT_PATH, sparse_matrix_train, fmt='%d')


# %%
# Building the test dataset
sparse_matrix_test = buildSparseMatrix(xtest, words_index, ytest)
sparse_matrix_test = sparse_matrix_test.groupby(
    ['DOC_ID', 'WORD_ID', 'LABEL']).sum()
sparse_matrix_test = sparse_matrix_test.reset_index()
print(sparse_matrix_test.shape)
print(sparse_matrix_test.head())
np.savetxt(TEST_MAT_PATH, sparse_matrix_test, fmt='%d')

# %%
