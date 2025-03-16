import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

model = tf.keras.models.load_model("face_detector_model.h5")
model.summary()

categories = ["angry", "happy", "neutral", "sad", "surprise"]

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

def predict_expression(face_img):
    if face_img.shape[0] == 0 or face_img.shape[1] == 0:
        return np.zeros(len(categories))  
    
    face_img = cv2.resize(face_img, (128, 128))
    face_img = np.expand_dims(face_img, axis=-1)  
    face_img = np.expand_dims(face_img, axis=0) 
    face_img = face_img / 255.0  
    predictions = model.predict(face_img)
    return predictions.flatten()

camera_index = 0  
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cv2.namedWindow("Expression Detector", cv2.WINDOW_NORMAL)  
cv2.namedWindow("Expression Probabilities", cv2.WINDOW_NORMAL)  

current_probabilities = np.zeros(len(categories))

while cap.isOpened():  
    ret, frame = cap.read()
    if not ret:
        continue  
    
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    updated_probabilities = np.zeros(len(categories))
    face_detected = False
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            face_img = frame[y:y+h_box, x:x+w_box]
            
            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                expression_probs = predict_expression(gray_face)
                updated_probabilities = np.maximum(updated_probabilities, expression_probs)
                face_detected = True
                expression = categories[np.argmax(expression_probs)]
                
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(frame, expression, (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2)
    
    if face_detected:
        current_probabilities = updated_probabilities
    else:
        current_probabilities *= 0.9  
    
    prob_display = np.zeros((300, 400, 3), dtype=np.uint8)
    y_offset = 50
    for i, category in enumerate(categories):
        prob_text = f"{category}: {current_probabilities[i]:.2f}"
        cv2.putText(prob_display, prob_text, (20, y_offset), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)
        y_offset += 40
    

    cv2.imshow("Expression Detector", frame)
    cv2.imshow("Expression Probabilities", prob_display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
