import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power((y_true-y_pred),2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred-y_true) / np.size(y_true)


def bce(y_true, y_pred):
    return np.negative(np.mean((y_true * np.log(y_pred)) + 
                               ((1-y_true) * np.log(1-y_pred))))

def bce_prime(y_true, y_pred):
    return (np.negative(np.divide(y_true,y_pred)) + 
            (np.divide((1-y_true),(1-y_pred))))


def cce(y_true,y_pred):
    return np.negative(np.mean(y_true * np.log(y_pred)))

def cce_prime(y_true,y_pred):
    return y_pred - y_true


# Test data for categorical classification (one-hot encoded targets)
y_true = np.array([[0, 1, 0], [1, 0, 0]])
y_pred = np.array([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]])

# Forward pass test
print("CCE:", cce(y_true, y_pred))

# Backward pass test
print("CCE Gradient:", cce_prime(y_true, y_pred))


