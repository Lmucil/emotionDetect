import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from cv2.dnn import blobFromImage

# Load the model
model = tf.keras.models.load_model("face_detector_model.h5")
model.summary()

categories = ["angry", "happy", "neutral", "sad", "surprise"]

# Load deep learning face detector
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel"
)

def predict_expression(face_img):
    face_img = cv2.resize(face_img, (128, 128))
    face_img = np.expand_dims(face_img, axis=-1)  
    face_img = np.expand_dims(face_img, axis=0) 
    face_img = face_img / 255.0  
    predictions = model.predict(face_img)
    return predictions.flatten()

camera_index = 0  
cap = cv2.VideoCapture(camera_index)

current_probabilities = np.zeros(len(categories))

def draw_heatmap(frame, probabilities):
    """Draws a heatmap on the camera frame."""
    heatmap = np.zeros((200, 400, 3), dtype=np.uint8)
    sns.heatmap([probabilities], annot=True, cmap="coolwarm", xticklabels=categories, yticklabels=[""], ax=plt.gca())
    plt.savefig("heatmap.png")
    heatmap_img = cv2.imread("heatmap.png")
    heatmap_img = cv2.resize(heatmap_img, (400, 100))
    frame[10:110, 10:410] = heatmap_img

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    blob = blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            face_img = frame[y:y1, x:x1]
            if face_img.size == 0:
                continue
            
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            expression_probs = predict_expression(gray_face)
            current_probabilities = expression_probs
            expression = categories[np.argmax(expression_probs)]
            
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, expression, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    draw_heatmap(frame, current_probabilities)
    cv2.imshow("Expression Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
