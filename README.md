# Spam vs Ham Classifier
 
Email spam filter built from scratch with NumPy (Foundations of Machine Learning, IIT Madras, Assignment 3). Two models are implemented and compared:
 
- **Perceptron**: linear, primal form, on L2-normalized bag-of-words with a bias term
- **Multinomial Naive Bayes**: Laplace smoothing, log-space MAP decision

## Dataset
 
[Enron-Spam](https://pages.aueb.gr/users/ion/data/enron-spam/readme.txt) (Metsis, Androutsopoulos & Paliouras, *Spam Filtering with Naive Bayes – Which Naive Bayes?*, CEAS 2006).
`enron1` and `enron2` are merged for training; 15 emails from `enron3` (10 ham, 5 spam) form the external test folder.
 
## Method
 
- **Features**: lowercase words matched by `[a-z]+`, a few stop-words removed (`subject`, `the`, `and`), counts of the 2000 most frequent words.
- **Perceptron**: each vector divided by its L2 norm, constant `1` appended, at most 20 epochs.
- **Naive Bayes**: raw counts, `log P(w|y) = log((N_yj + 1) / (N_y + V))`.
- **Validation**: random 85 % / 15 % train/validation split (seed 67).
## Results
 
| Model       | Validation accuracy | Test folder (`enron3`, 15 emails) |
|-------------|---------------------|-----------------------------------|
| Perceptron  | 98.85 %             | 1 false positive                  |
| Naive Bayes | 97.04 %             | 15/15 correct                     |
 
The test folder is very small, so its results are only indicative
 
## Structure
 
```
.
├── main.py      # preprocessing, training, validation, predict()
├── models.py    # Perceptron and Naive Bayes (from scratch)
├── data/
│   ├── ham/     # training ham emails (.txt)
│   └── spam/    # training spam emails (.txt)
├── test/        # emails to classify (.txt)
└── data.npz     # feature cache, created on first run
```
 
## Usage
 
Requires Python 3 and NumPy. The first run builds the vocabulary and caches the features in `data.npz`. Later runs reuse the cache.
 
Clean the test file and put other emails in .txt and run main.py : voilà
 
