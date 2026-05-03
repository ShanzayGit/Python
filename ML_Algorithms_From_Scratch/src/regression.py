import  numpy as np
def phi(x):
    return np.array([1, x])

# Basic train loss function for entire data
def trainLoss(w,training_data):
    return sum((np.dot(w, phi(x)) - y)**2 for x, y in training_data) / len(training_data)

# Basic gradiant loss function for entire data
def gradientTrainLoss(w,training_data):
    return sum(2 * (np.dot(w, phi(x)) - y) * phi(x) for x, y in training_data) / len(training_data)


def gradientDescent(w0,epochs=500, eta=0.1, training_data=None):
    w = w0.copy()
    eta = 0.1  
    for epoch in range(500):
        gradient = gradientTrainLoss(w, training_data)
        w -= eta * gradient
        print(f'Epoch {epoch}: w = {w}, Train Loss = {trainLoss(w, training_data)}')
    return w

# Train function (wrapper around gradient descent)
def train(training_data, epochs=500, eta=0.1):
    w0 = np.zeros(2)  # initial weights [bias, slope]
    w = gradientDescent(w0,epochs,eta,training_data)
    return w

# Predict function
def predict(X, w):
    return [np.dot(w, phi(x)) for x in X]
