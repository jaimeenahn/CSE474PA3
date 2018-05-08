import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def preprocess():
    """
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    new_train = np.concatenate((np.ones((n_data, 1)), train_data), 1)
    theta = sigmoid(np.sum(initialWeights.T * new_train, axis=1))
    theta = theta.reshape(n_data, -1)
    error = sum(labeli * np.log(theta) + (1 - labeli) * np.log(1 - theta)) * (-1 / n_data)
    error_grad = np.zeros((n_features + 1, 1))
    error_grad = (np.dot((theta - labeli).T, new_train) / n_data).T
    error_grad = error_grad.flatten()
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    new_X = np.concatenate((np.ones((data.shape[0], 1)), data), 1)
    for i in range(data.shape[0]):
        max = 0
        for j in range(10):
            tmp = sigmoid(np.dot(W[:,j], new_X[i]))
            if tmp > max:
                max = tmp
                label[i] = j

    return label


def mlrObjFunction(initialWeights_b, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.
    initialWeights_b = np.zeros((n_feature + 1, n_class))
    Input:
        initialWeights: the weight vector of size (D + 1) x k
        train_data: the data matrix of size N x D
        Y: the label vector of size N x k where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, Y = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    initialWeights_b = initialWeights_b.reshape(n_feature+1,10)
    n_class = initialWeights_b.shape[1]
    new_train = np.concatenate((np.ones((n_data, 1)), train_data), 1)

    ##################
    # YOUR CODE HERE #
    ##################
    theta = np.zeros((n_data, n_class))
 #   sumtheta = np.zeros((n_data, 1))
 #   for n in range(n_data):

    for j in range(initialWeights_b.shape[1]) :
        theta[:,j] = np.exp(np.sum((initialWeights_b[:,j].T * new_train), axis=1))
    #theta = np.exp(theta)
    theta = theta/(np.sum(theta, axis=1).reshape(n_data,-1))

    error = (-1)*np.sum(np.sum(Y*np.log(theta), axis=1), axis=0)

    error_grad = np.dot(new_train.T, (theta-Y)).ravel()
    # HINT: Do not forget to add the bias term to your input data
    print(error_grad)
    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    n_data = data.shape[0]

    ##################
    # YOUR CODE HERE #
    ##################
    new_X = np.concatenate((np.ones((data.shape[0], 1)), data), 1)
    theta = np.zeros((data.shape[0], W.shape[1]))
    for j in range(W.shape[1]):
        theta[:, j] = np.exp(np.sum((W[:, j].T * new_X), axis=1))
    theta = theta / (np.sum(theta, axis=1).reshape(n_data,-1))
    for n in range(data.shape[0]):
        max = 0
        for k in range(W.shape[1]):
            tmp = theta[n][k]
            if tmp > max:
                max = tmp
                label[n] = k

    return label



"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

"""
# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
"""
"""
Script for Support Vector Machine
"""
"""
print('\n\n--------------SVM-------------------\n\n')

svm = np.array(["linear", "rbf-default","rbf-0.1-1.0", "rbf-auto-10.0", "rbf-auto-20.0", "rbf-auto-30.0", "rbf-auto-40.0", "rbf-auto-50.0", "rbf-auto-60.0", "rbf-auto-70.0", "rbf-auto-80.0", "rbf-auto-90.0", "rbf-auto-100.0" ])
train_acc = np.array([])
validation_acc = np.array([])
test_acc = np.array([])
# C (1, 10, 20, 30, · · · , 100)
# linear, rbf
clf = SVC(kernel='linear')
clf.fit(train_data, train_label.ravel())
predicted_label = clf.predict(train_data)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label.ravel()).astype(float))) + '%')
train_acc = np.append(train_acc, 100 * np.mean((predicted_label == train_label.ravel()).astype(float)))
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label.ravel()).astype(float))) + '%')
validation_acc = np.append(validation_acc, 100 * np.mean((predicted_label == validation_label.ravel()).astype(float)))
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label.ravel()).astype(float))) + '%')
test_acc = np.append(test_acc, 100 * np.mean((predicted_label == test_label.ravel()).astype(float)))
print('\n\n--------------rbf-------------------\n\n')
#rbf-default
clf = SVC(kernel='rbf')
clf.fit(train_data, train_label.ravel())
predicted_label = clf.predict(train_data)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label.ravel()).astype(float))) + '%')
train_acc = np.append(train_acc, 100 * np.mean((predicted_label == train_label.ravel()).astype(float)))
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label.ravel()).astype(float))) + '%')
validation_acc = np.append(validation_acc, 100 * np.mean((predicted_label == validation_label.ravel()).astype(float)))
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label.ravel()).astype(float))) + '%')
test_acc = np.append(test_acc, 100 * np.mean((predicted_label == test_label.ravel()).astype(float)))
print('\n\n--------------gamma 0.1-------------------\n\n')
clf = SVC(kernel='rbf', gamma=0.1)
clf.fit(train_data, train_label.ravel())
predicted_label = clf.predict(train_data)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label.ravel()).astype(float))) + '%')
train_acc = np.append(train_acc, 100 * np.mean((predicted_label == train_label.ravel()).astype(float)))
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label.ravel()).astype(float))) + '%')
validation_acc = np.append(validation_acc, 100 * np.mean((predicted_label == validation_label.ravel()).astype(float)))
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label.ravel()).astype(float))) + '%')
test_acc = np.append(test_acc, 100 * np.mean((predicted_label == test_label.ravel()).astype(float)))
print('\n\n--------------C=10.0-------------------\n\n')
clf = SVC(kernel='rbf', C = 10.0)
clf.fit(train_data, train_label.ravel())
predicted_label = clf.predict(train_data)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label.ravel()).astype(float))) + '%')
train_acc = np.append(train_acc, 100 * np.mean((predicted_label == train_label.ravel()).astype(float)))
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label.ravel()).astype(float))) + '%')
validation_acc = np.append(validation_acc, 100 * np.mean((predicted_label == validation_label.ravel()).astype(float)))
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label.ravel()).astype(float))) + '%')
test_acc = np.append(test_acc, 100 * np.mean((predicted_label == test_label.ravel()).astype(float)))
print('\n\n--------------C=20.0-------------------\n\n')
clf = SVC(kernel='rbf', C = 20.0)
clf.fit(train_data, train_label.ravel())
predicted_label = clf.predict(train_data)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label.ravel()).astype(float))) + '%')
train_acc = np.append(train_acc, 100 * np.mean((predicted_label == train_label.ravel()).astype(float)))
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label.ravel()).astype(float))) + '%')
validation_acc = np.append(validation_acc, 100 * np.mean((predicted_label == validation_label.ravel()).astype(float)))
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label.ravel()).astype(float))) + '%')
test_acc = np.append(test_acc, 100 * np.mean((predicted_label == test_label.ravel()).astype(float)))
print('\n\n--------------C=30.0-------------------\n\n')
clf = SVC(kernel='rbf', C = 30.0)
clf.fit(train_data, train_label.ravel())
predicted_label = clf.predict(train_data)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label.ravel()).astype(float))) + '%')
train_acc = np.append(train_acc, 100 * np.mean((predicted_label == train_label.ravel()).astype(float)))
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label.ravel()).astype(float))) + '%')
validation_acc = np.append(validation_acc, 100 * np.mean((predicted_label == validation_label.ravel()).astype(float)))
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label.ravel()).astype(float))) + '%')
test_acc = np.append(test_acc, 100 * np.mean((predicted_label == test_label.ravel()).astype(float)))
print('\n\n--------------C=40.0-------------------\n\n')
clf = SVC(kernel='rbf', C = 40.0)
clf.fit(train_data, train_label.ravel())
predicted_label = clf.predict(train_data)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label.ravel()).astype(float))) + '%')
train_acc = np.append(train_acc, 100 * np.mean((predicted_label == train_label.ravel()).astype(float)))
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label.ravel()).astype(float))) + '%')
validation_acc = np.append(validation_acc, 100 * np.mean((predicted_label == validation_label.ravel()).astype(float)))
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label.ravel()).astype(float))) + '%')
test_acc = np.append(test_acc, 100 * np.mean((predicted_label == test_label.ravel()).astype(float)))
print('\n\n--------------C=50.0-------------------\n\n')
clf = SVC(kernel='rbf', C = 50.0)
clf.fit(train_data, train_label.ravel())
predicted_label = clf.predict(train_data)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label.ravel()).astype(float))) + '%')
train_acc = np.append(train_acc, 100 * np.mean((predicted_label == train_label.ravel()).astype(float)))
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label.ravel()).astype(float))) + '%')
validation_acc = np.append(validation_acc, 100 * np.mean((predicted_label == validation_label.ravel()).astype(float)))
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label.ravel()).astype(float))) + '%')
test_acc = np.append(test_acc, 100 * np.mean((predicted_label == test_label.ravel()).astype(float)))
print('\n\n--------------C=60.0-------------------\n\n')
clf = SVC(kernel='rbf', C = 60.0)
clf.fit(train_data, train_label.ravel())
predicted_label = clf.predict(train_data)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label.ravel()).astype(float))) + '%')
train_acc = np.append(train_acc, 100 * np.mean((predicted_label == train_label.ravel()).astype(float)))
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label.ravel()).astype(float))) + '%')
validation_acc = np.append(validation_acc, 100 * np.mean((predicted_label == validation_label.ravel()).astype(float)))
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label.ravel()).astype(float))) + '%')
test_acc = np.append(test_acc, 100 * np.mean((predicted_label == test_label.ravel()).astype(float)))
print('\n\n--------------C=70.0-------------------\n\n')
clf = SVC(kernel='rbf', C = 70.0)
clf.fit(train_data, train_label.ravel())
predicted_label = clf.predict(train_data)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label.ravel()).astype(float))) + '%')
train_acc = np.append(train_acc, 100 * np.mean((predicted_label == train_label.ravel()).astype(float)))
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label.ravel()).astype(float))) + '%')
validation_acc = np.append(validation_acc, 100 * np.mean((predicted_label == validation_label.ravel()).astype(float)))
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label.ravel()).astype(float))) + '%')
test_acc = np.append(test_acc, 100 * np.mean((predicted_label == test_label.ravel()).astype(float)))
print('\n\n--------------C=80.0-------------------\n\n')
clf = SVC(kernel='rbf', C = 80.0)
clf.fit(train_data, train_label.ravel())
predicted_label = clf.predict(train_data)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label.ravel()).astype(float))) + '%')
train_acc = np.append(train_acc, 100 * np.mean((predicted_label == train_label.ravel()).astype(float)))
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label.ravel()).astype(float))) + '%')
validation_acc = np.append(validation_acc, 100 * np.mean((predicted_label == validation_label.ravel()).astype(float)))
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label.ravel()).astype(float))) + '%')
test_acc = np.append(test_acc, 100 * np.mean((predicted_label == test_label.ravel()).astype(float)))
print('\n\n--------------C=90.0-------------------\n\n')
clf = SVC(kernel='rbf', C = 90.0)
clf.fit(train_data, train_label.ravel())
predicted_label = clf.predict(train_data)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label.ravel()).astype(float))) + '%')
train_acc = np.append(train_acc, 100 * np.mean((predicted_label == train_label.ravel()).astype(float)))
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label.ravel()).astype(float))) + '%')
validation_acc = np.append(validation_acc, 100 * np.mean((predicted_label == validation_label.ravel()).astype(float)))
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label.ravel()).astype(float))) + '%')
test_acc = np.append(test_acc, 100 * np.mean((predicted_label == test_label.ravel()).astype(float)))
print('\n\n--------------C=100.0-------------------\n\n')
clf = SVC(kernel='rbf', C = 100.0)
clf.fit(train_data, train_label.ravel())
predicted_label = clf.predict(train_data)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label.ravel()).astype(float))) + '%')
train_acc = np.append(train_acc, 100 * np.mean((predicted_label == train_label.ravel()).astype(float)))
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label.ravel()).astype(float))) + '%')
validation_acc = np.append(validation_acc, 100 * np.mean((predicted_label == validation_label.ravel()).astype(float)))
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label.ravel()).astype(float))) + '%')
test_acc = np.append(test_acc, 100 * np.mean((predicted_label == test_label.ravel()).astype(float)))

fig = plt.figure(figsize=[18,9])
plt.plot(svm,train_acc)
plt.plot(svm,validation_acc, 'r')
plt.plot(svm,test_acc, 'g')
plt.legend(['SVMs for Train Data', 'SVMs for Validation Data', 'SVMs for Test Data'])
plt.title('Relation between SVM models and accuracy')
plt.xlabel('hyperparameters')
plt.ylabel('accuracy')
plt.show()
"""
"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))
# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
