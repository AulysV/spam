#set page(margin: (x: 2cm, y: 2.2cm))
#set text(size: 11pt, font: "Libertinus Serif")
#set heading(numbering: "1.1")
#show raw: set text(font: "Fira Code", ligatures: true, weight: "semibold")

#show image: set image(width: 75%)

#align(center)[
  #text(size: 20pt, weight: "bold")[Foundations of Machine Learning \ Assignment 3]
  #v(0.3cm)
  Name : *Aulys VINAY* \
  Roll Number : *GE26Z833* \
  November 2026
  #v(0.3cm)
  #text(size: 15pt, weight: "bold")[SPAM vs HAM classificaiton]
]

#v(0.5cm)

= Objective

Binary classification of raw emails as SPAM or HAM.

= Raw data

After a few research on the Internet, the best data for this assignment seems to be the "Eron Spam" dataset (#link("https://pages.aueb.gr/users/ion/data/enron-spam/readme.txt")[→ readme ←]).

It originally contains 6 folders (from 6 eployees, classified as ham or spam). I took the 2 first data folders (eron1 and eron2) and combined them in the `data` folder as `ham` and `spam` in separate folders with the Power Rename tool from Microsoft Power Tools.

The test folder is 15 emails from eron3 (10 ham and 5 spams), to simulate the corrector's test.

= Choice of the algorithms

I chose to use Naive Bayes, wich seems to be the best algorithm seen in class to classify text.

But I will try to code a Perceptron first in order to see how to manipulate the data via Python. I will use a kernelised version for better accuracy.

KNN doesn't generate any models and decision trees seem complicated for word-type datasets.

= Using the data

The first step is making the data usable. I lost a bit of my familiarity with Pyton, as I mostly used MATLAB the few last years. I am seeing this Assignment as an opportunity to reimprove my skills.



= Kernel Perceptron

First step :

problème dimensions infines, donc on se contente du linéaire

= Naive Bayes

= Comparison with SVM

= Conclusion

