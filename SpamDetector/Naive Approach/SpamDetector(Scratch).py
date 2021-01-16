# %%
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# %%
# CONSTANTS
TRAIN_PROB_SPAM = 'SpamData/03_Testing/Train-SPAMPROB.txt'
TRAIN_PROB_HAM = 'SpamData/03_Testing/Train-HAMPROB.txt'
TRAIN_PROB_ALL = 'SpamData/03_Testing/Train-ALLPROB.txt'
TEST_FEATURES = 'SpamData/03_Testing/TEST-FEATURES.txt'
TEST_TARGET = 'SpamData/03_Testing/TEST-TARGET.txt'
SIZE = 2500
# From Probability calculation we know that ..
PROB_SPAM = 0.3116
# %%
# Loading the data
xtest = np.loadtxt(TEST_FEATURES, delimiter=' ')
ytest = np.loadtxt(TEST_TARGET, delimiter=' ')
prob_token_spam = np.loadtxt(TRAIN_PROB_SPAM, delimiter=' ')
prob_token_ham = np.loadtxt(TRAIN_PROB_HAM, delimiter=' ')
prob_token_all = np.loadtxt(TRAIN_PROB_ALL, delimiter=' ')
print(xtest.shape)


# %%
# Taking log of all values to maintain accuracy and reduce calculation complexity
joint_log_spam = xtest.dot(np.log(prob_token_spam))+np.log(PROB_SPAM)
joint_log_ham = xtest.dot(np.log(prob_token_ham))+np.log(1-PROB_SPAM)
predictions = joint_log_spam > joint_log_ham

# %%
# Metrics
# CAlculating Accuracy
correct_eval = (ytest == predictions).sum()
incorrect_eval = xtest.shape[0]-correct_eval
print(incorrect_eval)
print(correct_eval)
accuracy = correct_eval/len(xtest)*100
print(str(round(accuracy, 1))+" %")

# Visualizing data
ylabel = 'P(X|Spam)'
xlabel = 'P(X|HAM)'
plt.figure(figsize=(16, 7))
# Creating two grahs for better visualization
plt.subplot(1, 2, 1)
plt.xlabel(xlabel, fontsize=14)
plt.ylabel(ylabel, fontsize=14)
boundary = np.linspace(-14000, 1, 1000)
plt.scatter(joint_log_ham, joint_log_spam, alpha=0.5, s=25, color='navy')
plt.xlim([-14000, 1])
plt.ylim([-14000, 1])
plt.plot(boundary, boundary, color='orange')
plt.subplot(1, 2, 2)
plt.xlabel(xlabel, fontsize=14)
plt.ylabel(ylabel, fontsize=14)
boundary = np.linspace(-2000, 1, 1000)
plt.scatter(joint_log_ham, joint_log_spam, alpha=0.5, s=6, color='navy')
plt.xlim([-2000, 1])
plt.ylim([-2000, 1])
plt.plot(boundary, boundary, color='orange')
plt.show()

# %%
# Visualizing using seaborn
sns.set_style('whitegrid')
labels = 'Actual Category'
summary_df = pd.DataFrame(
    {ylabel: joint_log_spam, xlabel: joint_log_ham, labels: ytest})
sns.lmplot(x=xlabel, y=ylabel, data=summary_df, size=6.5, legend=False,
           fit_reg=False, scatter_kws={'alpha': 0.7, 's': 25}, hue=labels, palette='hls')
plt.xlim([-1000, 1])
plt.ylim([-1000, 1])
plt.plot(boundary, boundary, color='black')
plt.legend(('Decision Boundary', 'Non-spam', 'Spam'),
           loc='lower right', fontsize=14)
plt.show()

# %%
# Calculating other performance metrics
# Standard metrics
false_pos = ((ytest == 0) & (predictions == 1)).sum()
false_neg = ((ytest == 1) & (predictions == 0)).sum()
true_pos = ((ytest == 1) & (predictions == 1)).sum()
true_neg = ((ytest == 0) & (predictions == 0)).sum()

# Calculations based on metrics
recall_score = true_pos/(true_pos+false_neg)
precision_score = true_pos/(true_pos+false_pos)
f1_score = 2*(recall_score*precision_score)/(recall_score+precision_score)

# Results
print('True Positive : {:.2f}'.format(true_pos))
print('False Positive : {:.2f}'.format(false_pos))
print('True Negative: {:.2f}'.format(true_neg))
print('False Negative: {:.2f}'.format(false_neg))
print('Recall Score : {:.2f}'.format(recall_score))
print('Precision Score : {:.2f}'.format(precision_score))
print('F1-Score : {:.2f}'.format(f1_score))
