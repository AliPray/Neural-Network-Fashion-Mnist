#%%

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()

# Reshape data to 1D vector
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

# Normalize pixel values between 0 and 1
train_X = train_X / 255.0
test_X = test_X / 255.0

# Convert labels to one-hot vectors

num_classes = 10
train_y = np.eye(num_classes)[train_y]
test_y = np.eye(num_classes)[test_y]

# Set hyperparameters
learning_rate = 0.1
epochs = 10
batch_size = 128

# Initialize weights and bias
input_size = train_X.shape[1]
output_size = num_classes
W = np.random.randn(input_size, output_size) * 0.01
b = np.zeros((1, output_size))

# Define softmax activation function
def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Define cross-entropy loss function
def cross_entropy_loss(y_hat, y):
    m = y.shape[0]
    loss = -np.sum(y * np.log(y_hat)) / m
    return loss

# Define accuracy function
def accuracy(y_hat, y):
    pred = np.argmax(y_hat, axis=1)
    true = np.argmax(y, axis=1)
    acc = np.mean(pred == true)
    return acc

# Train model
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    # Shuffle data
    permutation = np.random.permutation(train_X.shape[0])
    train_X = train_X[permutation]
    train_y = train_y[permutation]

    # Mini-batch training
    for i in range(0, train_X.shape[0], batch_size):
        # Forward pass
        X_batch = train_X[i:i+batch_size]
        y_batch = train_y[i:i+batch_size]
        z = np.dot(X_batch, W) + b
        y_hat = softmax(z)

        # Backward pass
        dL = y_hat - y_batch
        dW = np.dot(X_batch.T, dL) / batch_size
        db = np.sum(dL, axis=0, keepdims=True) / batch_size
        W -= learning_rate * dW
        b -= learning_rate * db

    # Evaluate model on training and test sets
    z_train = np.dot(train_X, W) + b
    y_hat_train = softmax(z_train)
    train_loss.append(cross_entropy_loss(y_hat_train, train_y))
    train_acc.append(accuracy(y_hat_train, train_y))

    z_test = np.dot(test_X, W) + b
    y_hat_test = softmax(z_test)
    test_loss.append(cross_entropy_loss(y_hat_test, test_y))
    test_acc.append(accuracy(y_hat_test, test_y))

    #Print loss and accuracy
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss[-1]:.4f} - Train Acc: {train_acc[-1]:.4f} - Test Loss: {test_loss[-1]:.4f} - Test Acc: {test_acc[-1]:.4f}")


# Plot the loss and accuracy curves over epochs
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].plot(range(epochs), train_loss, 'r', label='Training Loss')
axs[0].plot(range(epochs), test_loss, 'b', label='Test Loss')
axs[0].set_title('Training and Test Loss')
axs[0].legend(loc='upper right')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')

axs[1].plot(range(epochs), train_acc, 'r', label='Training Accuracy')
axs[1].plot(range(epochs), test_acc, 'b', label='Test Accuracy')
axs[1].set_title('Training and Test Accuracy')
axs[1].legend(loc='lower right')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')

plt.show()


 #%%