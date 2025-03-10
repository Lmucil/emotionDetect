import numpy as np
import os
import cv2
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

trainDatasetPath = "dataset_FER_2013/train"
testDatasetPath = "dataset_FER_2013/test"

categories = ["angry", "happy", "neutral", "sad", "surprise"]
numClasses = len(categories)
imageSize = (128, 128)  

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
                print(imgPath + ": image loaded successfully")
    return np.array(X, dtype=np.float32), np.array(y).reshape(-1, 1)

XTrain, yTrain = load_images_from_folder(trainDatasetPath)

XTest, yTest = load_images_from_folder(testDatasetPath)


encoder = OneHotEncoder(sparse_output=False)
yTrain = encoder.fit_transform(yTrain)
yTest = encoder.transform(yTest)

print("Training data shape:", XTrain.shape, yTrain.shape)
print("Testing data shape:", XTest.shape, yTest.shape)

print("Training data distribution:", Counter(yTrain.argmax(axis=1)))
print("Testing data distribution:", Counter(yTest.argmax(axis=1)))

inputSize = XTrain.shape[1]
hiddenSize = 150
outputSize = numClasses
learningRate = 0.0005
epochs = 500


np.random.seed(42)
W1 = np.random.randn(inputSize, hiddenSize) * np.sqrt(1.0 / inputSize)
W2 = np.random.randn(hiddenSize, outputSize) * np.sqrt(1.0 / hiddenSize)
b1 = np.zeros((1, hiddenSize))
b2 = np.zeros((1, outputSize))


def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-8)



def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

def computeLoss(yTrue, yPred):
    epsilon = 1e-8  
    return -np.mean(yTrue * np.log(yPred + epsilon))


for epoch in range(epochs):
    Z1 = np.dot(XTrain, W1) + b1
    A1 = leaky_relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    
    loss = computeLoss(yTrain, A2)

    dZ2 = A2 - yTrain
    dW2 = np.dot(A1.T, dZ2) / len(XTrain)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / len(XTrain)
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * leaky_relu_derivative(Z1)
    dW1 = np.dot(XTrain.T, dZ1) / len(XTrain)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / len(XTrain)
    
    W1 -= learningRate * dW1
    b1 -= learningRate * db1
    W2 -= learningRate * dW2
    b2 -= learningRate * db2
    
if epoch % 1 == 0:
    print(f"Epoch {epoch}, Loss: {loss:.4f}, Predicted: {A2[:5]}")


Z1_test = np.dot(XTest, W1) + b1
A1_test = leaky_relu(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
A2_test = softmax(Z2_test)

predictions = np.argmax(A2_test, axis=1)
true_labels = np.argmax(yTest, axis=1)

np.savez("face_detector_model.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("Model training complete, 'face_detector_model.npz'")

accuracy = np.mean(predictions == true_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

