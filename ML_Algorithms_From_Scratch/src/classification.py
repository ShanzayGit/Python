import numpy as np
def phi(x):
    return np.array(x)

def initialWeightVector():
    return np.zeros(2)

def trainLoss(w,X_train,y_train):
    return 1.0 / len(X_train) * sum(
        max(1 - (w.dot(phi(x))) * y, 0) for x, y in zip(X_train, y_train)
    )

def gradientTrainLoss(w,X_train,y_train):

    traingradient = np.zeros(2)
    for x, y in zip(X_train, y_train):
        if (1 - w.dot(phi(x))*y) > 0:
            gradient = -phi(x)*y
        else:
            gradient = np.zeros(2)
        traingradient += gradient

    return traingradient / len(X_train)
    
def predict(x, w):
    value = w.dot(phi(x))
    if value == 0:
        return 0
    elif value>0:
        return 1
    else:
        return -1

# Gradient Descent Training
losses=[]
boundaries=[]
def gradientDescent(X_train,y_train):
    w = initialWeightVector()
    eta = 0.1
    for t in range(500):
        value = trainLoss(w, X_train, y_train)
        gradient = gradientTrainLoss(w, X_train, y_train)
        w = w - eta * gradient
        #save history
        losses.append(value)
        boundaries.append(w.copy())
        print(f'epoch {t}: w = {w}, TrainLoss = {value}, gradientTrainLoss = {gradient}')
    return w

