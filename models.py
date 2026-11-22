import numpy as np

# Kernel Perceptron (bad ida so removed it)

# def kernel(X1, X2, tipe="polynomial", deg=2, gamma=0.1): # type was special function so tipe
#     if tipe == "polynomial":
#         return (np.dot(X1, X2)+1)**deg
#     if tipe == "rbf":
#         return np.exp(-gamma * np.sum((X1 - X2) ** 2))

def perceptron(X, y, max_aliens=20): # 20 max wrongly classed points (20 arbitrarely found to be the best)

    w = np.zeros(X.shape[1])

    for a in range(max_aliens):
        errors = 0
        for i in range(len(X)):  # for each email i
            if y[i] * np.dot(w, X[i]) <= 0:
                w = w + y[i] * X[i]
                errors += 1

        if errors == 0:  # convergence
            break

    return w

def perceptron_test(X, w):
    return np.where(X @ w > 0, 1, -1)


def bayes_train(X, y):

    # log priors 
    pone = np.log(np.mean(y == 1))
    mone = np.log(np.mean(y == -1))

    # log likelihoods 
    count_pone = np.sum(X[y == 1], axis=0) + 1
    like_pone = np.log(count_pone / np.sum(count_pone))

    count_mone = np.sum(X[y == -1], axis=0) + 1
    like_mone = np.log(count_mone / np.sum(count_mone))

    return pone, mone, like_pone, like_mone

def bayes_test(X, param):

    pone, mone, like_pone, like_mone = param

    # y = argmax (logprior + sum xloglikelihoods)
    isspam = pone + X @ like_pone
    isham = mone + X @ like_mone
    # y = np.sign(isham - isspam)
    y = np.where(isspam > isham, 1, -1)

    return  y