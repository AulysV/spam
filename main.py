import numpy as np
import os
import re

import models

"""
Raw data : ERON-SPAM

V. Metsis, I. Androutsopoulos and G. Paliouras, "Spam Filtering with 
Naive Bayes - Which Naive Bayes?". Proceedings of the 3rd Conference 
on Email and Anti-Spam (CEAS 2006), Mountain View, CA, USA, 2006.
"""

# Variables : 

# vocab_total : list of all words (lowercase, letters) in all emails
# expulse : words that are not useful for the analysis
# all_emails : list of list of words of every email
# y_list : temp label list (+1 and -1)
# X : numpy array line i is email i and X[i, j] is how many times the word j appears in i
# y : numpy array of labels
# vocab : dictionnaire that translates email words into datapoints 

ham_files = os.listdir("data/ham")
spam_files = os.listdir("data/spam")

vocab_total = {} # dictionnary with how often a word appears 

expulse = { # words that are in every mail
    "subject", "the", "and", "for", "that", "with", "you", "are", 
    "this", "from", "have", "your", "not", "but", "was", "were", "can"
}

# print(ham_files+spam_files)

if not(os.path.exists("data.npz")): 
    # storing data not to do the scanning of the mails again using numpy npz files

    all_emails = []
    y_list = [] 

    for file in ham_files:

        with open("data/ham/"+file, "r", encoding="utf-8", errors="ignore") as f: email = f.read()
        email_words = re.findall(r"[a-z]+", email.lower()) # email : ["list", "of", "strings"]

        all_emails.append(email_words)
        y_list.append(-1)

        for mot in email_words:
            if mot in expulse:
                continue
            if mot in vocab_total:
                vocab_total[mot] += 1
            else:
                vocab_total[mot] = 1

    for file in spam_files:

        with open("data/spam/"+file, "r", encoding="utf-8", errors="ignore") as f: email = f.read()
        email_words = re.findall(r"[a-z]+", email.lower()) # email : ["list", "of", "strings"]

        all_emails.append(email_words)
        y_list.append(1)

        for mot in email_words:
            if mot in expulse:
                continue
            if mot in vocab_total:
                vocab_total[mot] += 1
            else:
                vocab_total[mot] = 1

    sorted_words = sorted(vocab_total.items(), key=lambda x: x[1], reverse=True)[:2000]
    vocab = {mot: idx for idx, (mot, _) in enumerate(sorted_words)} # 2000 first most used words

    X = np.zeros((len(all_emails), len(vocab)))
    y = np.array(y_list)

    for i, mots in enumerate(all_emails):
        for mot in mots:
            if mot in vocab:
                X[i, vocab[mot]] += 1

    np.savez("data.npz", X=X, y=y, mots=np.array(list(vocab.keys())))

    print(vocab_total)
    print(len(vocab_total))

else:

    data = np.load("data.npz")
    X = data["X"]
    y = data["y"]
    vocab = {mot: i for i, mot in enumerate(data["mots"])}



# =============== Perceptron

# Normalisation (because we are going to bias trick):

norms = np.linalg.norm(X, axis=1, keepdims=True)
norms[norms == 0] = 1 # TO avoid dividing by 0
Xn = X / norms

# print(X.shape)

# Bias trick to have affine perceptron

Xn = np.hstack((Xn, np.ones((Xn.shape[0], 1))))
yn = y # because we are reordering for Bayes and it made it crash (i put a "n" to match Xn)

# Randomisation and then splitting into 80% training 20% test

np.random.seed(67)
indices = np.random.permutation(len(X))
Xn, yn = Xn[indices], yn[indices]

split = int(0.85 * len(Xn))
X_train, y_train = Xn[:split], yn[:split]
X_val, y_val = Xn[split:], yn[split:]

# print(X)

w = models.perceptron(X_train, y_train)
# print(w)

# Test on the validation data

side = X_val @ w
y_predicted = np.sign(side)

accuracy = np.mean(y_predicted == y_val)

print(f"Accuracy of Perceptron on validation data : {round(accuracy*100, 2)}%")




# =============== Naive Bayes

# np.random.seed(69)
# indices = np.random.permutation(len(Xb))
# Xb, y = Xb[indices], y[indices]

# split = int(0.85 * len(Xn))
# X_train2, y_train2 = Xb[:split], y[:split]
# X_val2, y_val2 = Xb[split:], y[split:]

X_train2, y_train2 = X[indices[:split]], y[indices[:split]]
X_val2, y_val2     = X[indices[split:]], y[indices[split:]]

params_bayes = models.bayes_train(X_train2, y_train2)
y_predicted2 = models.bayes_test(X_val2, params_bayes)

accuracy2 = np.mean(y_predicted2 == y_val2)

print(f"Accuracy of Bayes on validation data : {round(accuracy2*100, 2)}%")

