import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

trainDatasetPath = "dataset_FER_2013/train"
testDatasetPath = "dataset_FER_2013/test"

categories = ["angry", "happy", "neutral", "sad", "surprise"]
numClasses = len(categories)
imageSize = (48, 48)

def load_images_from_folder(folderPath):
    X, y = [], []
    for label, category in enumerate(categories):
        categoryPath = os.path.join(folderPath, category)
        for filename in os.listdir(categoryPath):
            imgPath = os.path.join(categoryPath, filename)
            img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, imageSize) / 255.0
                X.append(img.flatten())
                y.append(label)
    return np.array(X, dtype=np.float32), np.array(y).reshape(-1, 1)

XTrain, yTrain = load_images_from_folder(trainDatasetPath)
XTest, yTest = load_images_from_folder(testDatasetPath)

encoder = OneHotEncoder(sparse_output=False)
yTrain = encoder.fit_transform(yTrain)
yTest = encoder.transform(yTest)

inputSize = XTrain.shape[1]
hiddenSize = 128
outputSize = numClasses

np.random.seed(42)
W1 = np.random.randn(inputSize, hiddenSize) * np.sqrt(2.0 / inputSize)
W2 = np.random.randn(hiddenSize, outputSize) * np.sqrt(2.0 / hiddenSize)
b1 = np.zeros((1, hiddenSize))
b2 = np.zeros((1, outputSize))

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def reluDerivative(z):
    return (z > 0).astype(float)

def computeLoss(yTrue, yPred):
    return -np.mean(yTrue * np.log(np.clip(yPred, 1e-8, 1)))


def learning_rate_finder(X, y, min_lr=1e-7, max_lr=1, num_steps=10, epochs=3):
    global W1, W2, b1, b2  

    learning_rates = np.logspace(np.log10(min_lr), np.log10(max_lr), num_steps)
    losses = []

    for lr in learning_rates:
        W1 = np.random.randn(inputSize, hiddenSize) * np.sqrt(2.0 / inputSize)
        W2 = np.random.randn(hiddenSize, outputSize) * np.sqrt(2.0 / hiddenSize)
        b1 = np.zeros((1, hiddenSize))
        b2 = np.zeros((1, outputSize))

        # Train for a few epochs
        for epoch in range(epochs):
            Z1 = np.dot(X, W1) + b1
            A1 = relu(Z1)
            Z2 = np.dot(A1, W2) + b2
            A2 = softmax(Z2)

            loss = computeLoss(y, A2)

            # Backpropagation
            dZ2 = A2 - y
            dW2 = np.dot(A1.T, dZ2) / len(X)
            db2 = np.sum(dZ2, axis=0, keepdims=True) / len(X)

            dA1 = np.dot(dZ2, W2.T)
            dZ1 = dA1 * reluDerivative(Z1)
            dW1 = np.dot(X.T, dZ1) / len(X)
            db1 = np.sum(dZ1, axis=0, keepdims=True) / len(X)

            # Update weights
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        # Store final loss for this learning rate
        losses.append(loss)
        print(f"LR: {lr:.8f}, Loss: {loss:.4f}")

    # Plot loss vs learning rate
    plt.figure(figsize=(8, 6))
    plt.plot(learning_rates, losses, marker="o")
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.grid(True)
    plt.show()

# Run Learning Rate Finder
learning_rate_finder(XTrain, yTrain)
