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
  #text(size: 15pt, weight: "bold")[SPAM vs HAM classification]
]

#v(0.5cm)

= Objective

Binary classification of raw emails as SPAM or HAM. Labels are $y in {-1, +1}$ during training (HAM $= -1$, SPAM $= +1$) and are mapped to ${0, 1}$ (non-spam, spam) in the final output, as required by the assignment.

= Raw data

After some research on the Internet, the most suitable data for this assignment seemed to be the *Enron-Spam* dataset (Metsis, Androutsopoulos and Paliouras, "Spam Filtering with Naive Bayes -- Which Naive Bayes?", CEAS 2006, #link("https://pages.aueb.gr/users/ion/data/enron-spam/readme.txt")[→ readme ←]). It is publicly available and already labelled.

It originally contains 6 folders (one per Enron employee), each split into ham and spam emails. I took the first two (`enron1` and `enron2`) and merged them into a single `data` folder with two subfolders, `ham` and `spam`, using the PowerRename tool from Microsoft PowerToys.

// TODO: fill in the exact counts (e.g. print(len(ham_files), len(spam_files)))
Resulting training pool: *N_ham* ham emails and *N_spam* spam emails (replace with the exact counts, to show how balanced the classes are).

*Validation protocol.* The pool is randomly shuffled (NumPy seed `67`) and split into *85 % training / 15 % validation*. In addition, the `test` folder contains some random emails taken from `enron3`, a employee never seen during training, to simulate the your (the corrector) test set.

= Choice of the algorithms

I chose to implement a *Perceptron* (discriminative) and a *Multinomial Naive Bayes* (generative), so that two very different families can be compared on the same features.

I first considered a kernel to make the Perceptron non-linear (see @kernel). Other algorithms were discarded: $k$-NN does not produce a compact model (every prediction requires scanning the whole training set), and decision trees seemed poorly suited to sparse word-count vectors in dimension $2000$.

= Using the data

The first step is making the raw emails usable. I lost a bit of my familiarity with Python, as I mostly used MATLAB the last few years. I am seeing this assignment as an opportunity to improve my skills again.

Each email is lowercased and split into alphabetic words with the regular expression `[a-z]+`. A few uninformative words present in practically every email are removed (`subject`, `the`, `and` (and other more were planned to be removed but it didn't do anything on the accuracy)).

I keep the 2000 most frequent words of the training corpus as vocabulary. Each email $i$ becomes a vector $x_i in RR^(2000)$ where $x_(i j)$ is the number of occurrences of word $j$ in email $i$. The resulting matrix $X in RR^(N times 2000)$ is cached in `data.npz`, so that the text files only need to be scanned once.

*Model-specific preprocessing.*
- *Perceptron:* each vector is normalized, $tilde(x)_i = x_i / norm(x_i)_2$, to remove the effect of email length and bound the radius of the data (emails with a zero norm are left unchanged). I then apply the *bias trick*: a constant component $1$ is appended (dimension $2001$), which lets the Perceptron learn an affine decision boundary without handling a bias separately
- *Naive Bayes:* raw integer counts are kept (no normalization, no bias)

= Kernel Perceptron <kernel>

My first idea was to use a kernel Perceptron, in case the data were not linearly separable. It turned out to be a bad idea:
- An RBF kernel corresponds to an infinite-dimensional feature space, so it can only be used in the dual form
- Even a degree-2 polynomial kernel on $2000$ words would need about $2001 dot 2002 / 2 approx 2 times 10^6$ explicit features, which does not fit in memory

= Naive Bayes

The  Naive Bayes assumes that, given the class, the words of an email are conditionally independent. With vocabulary size 2000, the parameters are estimated by maximum likelihood with +1 smoothies, to avoid zero probabilities for words never seen in one class.
And then as seen in class we have :

$ hat(y) = op("arg max", limits: #true)_(y in {-1, +1}) (log P(y) + sum_(j=1)^V x_j log P(w_j | y)) . $

This is a single matrix-vector product per class, so prediction is super fast.

= Implementatioon

The code is split in two files:
- `models.py` contains the algorithms, coded from scratch with NumPy only for array operations (`perceptron`, `bayes_train`, `bayes_test`).
- `main.py` handles preprocessing, the train/validation split, training of both models and the prediction on the `test` folder. The function `predict(folder="test")` reads every email file of the folder, builds the feature vectors with the training vocabulary, and returns for each file the prediction of both models, with 1 = spam and 0 = non-spam.

*Results :*

Percepton : *98.85%*, with one false positive and one false negative on the test (eron3) folder.

Naive Bayes : *97.04%* and all correct on the test (eron3) folder


On the validation set the Perceptron is better. On the emails of `enron3`, Naive Bayes classifies everything correctly.

// TODO: add the hyperparameter plot once computed (axes must be labelled).
// Example: validation accuracy (%) vs. maximum number of epochs (to justify max_epochs = 20),
// or vs. vocabulary size V in {500, 1000, 2000, 4000}.
// #figure(
//   image("accuracy_vs_epochs.png"),
//   caption: [Validation accuracy of the Perceptron as a function of the maximum number of epochs.],
// )

= Conclusion

Both approaches exceed $97 %$ accuracy with a light pipeline (bag-of-words on 2000 words, no external library for the algorithms). The Perceptron has the best validation accuracy, while Naive Bayes behaved perfectly on the kind of out of training data `enron3` sample.

Given the small size of that sample, I would not conclude that one model is better in general. The choice of the final output model is made from the validation accuracy and discussed above.
