import numpy as np
from tensorflow.keras.datasets import mnist

#Initializing weights and biases
def initialize_weights():
    w1 = np.random.randn(784, 128) * np.sqrt(1. / 784)
    b1 = np.zeros((1, 128))

    w2 = np.random.randn(128, 64) * np.sqrt(1. / 128)
    b2 = np.zeros((1, 64))

    w3 = np.random.randn(64, 10) * np.sqrt(1. / 64)
    b3 = np.zeros((1, 10))
    return w1, b1, w2, b2, w3, b3
    
def one_hot_encode(y_train, y_test, num_classes):
    Y_TRAIN = np.eye(num_classes)[y_train]
    Y_TEST = np.eye(num_classes)[y_test]
    return Y_TRAIN, Y_TEST

def forwardpass(X, w1, b1, w2, b2, w3, b3):
    z1 = (X @ w1)
    z1 += b1
    a1 = relu(z1)

    z2 = (a1 @ w2)
    z2 += b2
    a2 = relu(z2)

    z3 = (a2 @ w3)
    z3 += b3
    y = softmax(z3)
    return y, a2, a1, z2, z1

def relu(z):
    return np.maximum(0,z)
        

def softmax(z):
    result = np.zeros_like(z)
    for i in range(z.shape[0]):
        exp_row = (np.exp(z[i] - np.max(z[i]))) 
        result[i] = exp_row / np.sum(exp_row)
    return result

def crossentropy_loss(predicted, actual):
    dz3 = predicted - actual
    return dz3

def backpropagation(X, dz3, w3, w2, a2, a1, z2, z1):
    dw3 = a2.T @ dz3
    db3 = np.sum(dz3, axis=0, keepdims=True)

    da2 = dz3 @ w3.T
    dz2 = da2 * (z2 > 0)
    dw2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2 @ w2.T
    dz1 = da1 * (z1 > 0)
    dw1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    return dw3, db3, dw2, db2, dw1, db1

def update_parameters(w1, b1, w2, b2, w3, b3, dw3, db3, dw2, db2, dw1, db1, learning_rate):
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    w3 = w3 - learning_rate * dw3
    b3 = b3 - learning_rate * db3
    return w1, b1, w2, b2, w3, b3




def main():
    #loading the mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #reshaping the training and test data to be a 1d 784 vector with each element being 1 pixel
    X_TRAIN = np.reshape(x_train, (60000, 784))
    X_TEST = np.reshape(x_test, (10000, 784))

    #normalized testing data to be between 0 and 1
    normalized_training_data = X_TRAIN / 255
    normalized_testing_data = X_TEST / 255

    #reshaping the labels so that they are not the individual numbers(1 or 3 or 5 etc) but instead
    #they are a 1d 10 element vector with the numbers represented using one hot encoding
    num_classes = np.max(y_train) + 1
    Y_TRAIN, Y_TEST = one_hot_encode(y_train, y_test, num_classes)

    w1, b1, w2, b2, w3, b3, = initialize_weights()
    batch_size = 64
    epochs = 20
    for epoch in range(epochs):
        batch_losses = []
        total_correct = 0
        total_samples = 0
        for i in range(0, len(normalized_training_data), batch_size):
            X = normalized_training_data[i:i + batch_size]
            actual = Y_TRAIN[i:i + batch_size]
            predicted, a2, a1, z2, z1 = forwardpass(X, w1, b1, w2, b2, w3, b3)
            dz3 = crossentropy_loss(predicted, actual)
            predictions = np.argmax(predicted, axis=1)
            true_labels = np.argmax(actual, axis=1)
            correct_predictions = np.sum(predictions == true_labels)
            total_correct += correct_predictions
            total_samples += batch_size
            epsilon = 1e-8
            log_predictions = np.log(predicted + epsilon)
            batch_loss = -np.mean(np.sum(actual * log_predictions, axis=1))
            batch_losses.append(batch_loss)
            dw3, db3, dw2, db2, dw1, db1 = backpropagation(X, dz3, w3, w2, a2, a1, z2, z1)
            learning_rate = 0.01
            w1, b1, w2, b2, w3, b3 = update_parameters(w1, b1, w2, b2, w3, b3, dw3, db3, dw2, db2, dw1, db1, learning_rate)
        epoch_loss = np.mean(batch_losses)
        accuracy = total_correct / total_samples
        print("Epoch: " + str(epoch+1), "Loss: " + str(epoch_loss), "Accuracy: " + str(accuracy))

    correct = 0
    batch_size = 64
    for i in range(0, len(normalized_testing_data), batch_size):
        X = normalized_testing_data[i:i + batch_size]
        actual = Y_TEST[i:i + batch_size]
        predicted, a2, a1, z2, z1 = forwardpass(X, w1, b1, w2, b2, w3, b3)
        predictions = np.argmax(predicted, axis=1)
        true_labels = np.argmax(actual, axis=1)
        correct += np.sum(predictions == true_labels)
    test_accuracy = correct / len(normalized_testing_data)
    print("Test Accuracy: " + str(test_accuracy))





            
        














if __name__ == "__main__":
    main()