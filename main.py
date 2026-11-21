import numpy as np
import os
import re

"""
Raw data : ERON-SPAM

V. Metsis, I. Androutsopoulos and G. Paliouras, "Spam Filtering with 
Naive Bayes - Which Naive Bayes?". Proceedings of the 3rd Conference 
on Email and Anti-Spam (CEAS 2006), Mountain View, CA, USA, 2006.
"""

ham_files = os.listdir("data/ham")
spam_files = os.listdir("data/spam")

vocab_total = {} # dictionnary with how often a word appears 

expulse = { # words that are in every mail
    "subject", "the", "and", "for", "that", "with", "you", "are", 
    "this", "from", "have", "your", "not", "but", "was", "were", "can"
}

# print(ham_files+spam_files)

if not(os.path.exists("data.npz")):

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
    vocab = {mot: idx for idx, (mot, _) in enumerate(sorted_words)}

    X = np.zeros((len(all_emails), len(vocab)), dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

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


# Perceptron