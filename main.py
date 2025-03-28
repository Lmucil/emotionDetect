import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import threading
from collections import Counter

model_data = np.load("expression_detector_model.npz")
W1, b1, W2, b2 = model_data["W1"], model_data["b1"], model_data["W2"], model_data["b2"]


categories = ["angry", "happy", "neutral", "sad", "surprise"]

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def relu(z):
    return np.maximum(0, z)

def predict_expression(face_img):
    face_img = cv2.resize(face_img, (48, 48)).reshape(1, -1) / 255.0
    Z1 = np.dot(face_img, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return A2.flatten()  

camera_index = 0  
cap = cv2.VideoCapture(camera_index)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

current_probabilities = np.zeros(len(categories))

def update_graph():
    """Function to update the graph in real-time"""
    global current_probabilities
    plt.ion()  

    while True:
        plt.clf()
        ax = plt.gca()
        ax.set_ylim([0, 1])  
        ax.set_ylabel("Probability")
        ax.set_title("Expression Probabilities")

        plt.bar(categories, current_probabilities, color=['red', 'yellow', 'blue', 'purple', 'orange'])

        sns.heatmap([current_probabilities], annot=True, cmap="coolwarm", xticklabels=categories, yticklabels=[""])
        
        plt.pause(0.1)  

graph_thread = threading.Thread(target=update_graph, daemon=True)
graph_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        expression_probs = predict_expression(face_img)
        current_probabilities = expression_probs  
        expression = categories[np.argmax(expression_probs)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, expression, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Expression Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
