import numpy as np

# Kernel Perceptron (bad ida so removed it)

# def kernel(X1, X2, tipe="polynomial", deg=2, gamma=0.1): # type was special function so tipe
#     if tipe == "polynomial":
#         return (np.dot(X1, X2)+1)**deg
#     if tipe == "rbf":
#         return np.exp(-gamma * np.sum((X1 - X2) ** 2))

def perceptron(X, y, max_aliens=50): # 50 max wrongly classed points

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