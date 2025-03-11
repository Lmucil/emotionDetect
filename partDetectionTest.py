import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import threading
from tensorflow.keras.utils import get_custom_objects

# Load the trained Keras model
model = tf.keras.models.load_model("face_emotion_model.keras")

# Emotion categories (must match trained model)
categories = ["angry", "happy", "neutral", "sad", "surprise"]

# Open webcam
camera_index = 0
cap = cv2.VideoCapture(camera_index)

# Load face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Graph update settings
current_probabilities = np.zeros(len(categories))

def update_graph():
    """Function to update the graph in real-time"""
    global current_probabilities
    plt.ion()  # Enable interactive mode

    while True:
        plt.clf()
        ax = plt.gca()
        ax.set_ylim([0, 1])  
        ax.set_ylabel("Probability")
        ax.set_title("Expression Probabilities")

        plt.bar(categories, current_probabilities, color=['red', 'yellow', 'blue', 'purple', 'orange'])

        sns.heatmap([current_probabilities], annot=True, cmap="coolwarm", xticklabels=categories, yticklabels=[""])
        
        plt.pause(0.1)  # Refresh every 100ms

# Start graph updating in a separate thread
graph_thread = threading.Thread(target=update_graph, daemon=True)
graph_thread.start()

def predict_expression(face_img):
    face_img = cv2.resize(face_img, (128, 128)) / 255.0
    face_img = face_img.reshape(1, 128, 128, 1)

    # Extract left and right eyes separately
    left_eye = face_img[:, 20:60, 20:60, :]
    right_eye = face_img[:, 20:60, 68:108, :]
    
    # Resize eyes and stack them along axis 1 to match (batch, 2, 32, 32, 1)
    left_eye = cv2.resize(left_eye[0], (32, 32)).reshape(1, 32, 32, 1)
    right_eye = cv2.resize(right_eye[0], (32, 32)).reshape(1, 32, 32, 1)
    eyes_img = np.stack([left_eye, right_eye], axis=1)  # Shape becomes (1, 2, 32, 32, 1)

    # Extract and reshape mouth region (ensure it's (batch, 64, 32, 1))
    mouth_img = face_img[:, 80:144, 40:72, :]  # 64x32 pixels region
    mouth_img = cv2.resize(mouth_img[0], (32, 64)).reshape(1, 64, 32, 1)  # (batch, 64, 32, 1)

    # Pass all three inputs correctly
    predictions = model.predict([face_img, eyes_img, mouth_img])  
    return predictions.flatten()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        expression_probs = predict_expression(face_img)
        current_probabilities = expression_probs  # Update graph

        expression = categories[np.argmax(expression_probs)]  # Get best prediction

        # Draw face box and text
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, expression, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Expression Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
