import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Reshape

# Define Dataset Paths
trainDatasetPath = "dataset_FER_2013/train"
testDatasetPath = "dataset_FER_2013/test"

# Emotion Categories
categories = ["angry", "happy", "neutral", "sad", "surprise"]
numClasses = len(categories)

# Image Sizes for Different Features
faceSize = (128, 128)
eyeSize = (32, 32)
mouthSize = (64, 32)

# Load OpenCV Haar Cascades for Face, Eyes, and Mouth
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

def extract_face_parts(image):
    """Detects and extracts the face, eyes, and mouth from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Extract face region
        face_resized = cv2.resize(face, faceSize) / 255.0

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(face)
        eye_images = []
        for (ex, ey, ew, eh) in eyes[:2]:  # Consider first two detected eyes
            eye = face[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, eyeSize) / 255.0
            eye_images.append(eye)

        # Ensure we have two eyes (if not, pad with blank images)
        while len(eye_images) < 2:
            eye_images.append(np.zeros(eyeSize))

        # Detect mouth (lower part of the face)
        mouths = mouth_cascade.detectMultiScale(face, 1.7, 20)
        mouth_image = np.zeros(mouthSize)  # Default blank image
        for (mx, my, mw, mh) in mouths:
            # Ensure mouth is in lower half of the face
            if my > h // 2:
                mouth = face[my:my+mh, mx:mx+mw]
                mouth_image = cv2.resize(mouth, mouthSize) / 255.0
                break

        return face_resized, np.array(eye_images), mouth_image

    return None, None, None  # No face detected

def load_images_with_features(folderPath):
    """Loads images and extracts face, eye, and mouth features for training/testing."""
    X_face, X_eyes, X_mouth, y = [], [], [], []
    for label, category in enumerate(categories):
        categoryPath = os.path.join(folderPath, category)
        for filename in os.listdir(categoryPath):
            imgPath = os.path.join(categoryPath, filename)
            img = cv2.imread(imgPath)
            if img is not None:
                face, eyes, mouth = extract_face_parts(img)
                if face is not None and eyes is not None and mouth is not None:
                    X_face.append(face)
                    X_eyes.append(eyes)
                    X_mouth.append(mouth)
                    y.append(label)
    
    return np.array(X_face, dtype=np.float32), np.array(X_eyes, dtype=np.float32), np.array(X_mouth, dtype=np.float32), np.array(y)

# Load Train & Test Data
XFaceTrain, XEyesTrain, XMouthTrain, yTrain = load_images_with_features(trainDatasetPath)
XFaceTest, XEyesTest, XMouthTest, yTest = load_images_with_features(testDatasetPath)

# Convert Labels to Categorical
yTrain = to_categorical(yTrain, numClasses)
yTest = to_categorical(yTest, numClasses)

# Reshape Data for CNN
XFaceTrain = XFaceTrain.reshape(-1, faceSize[0], faceSize[1], 1)
XFaceTest = XFaceTest.reshape(-1, faceSize[0], faceSize[1], 1)
XEyesTrain = XEyesTrain.reshape(-1, 2, eyeSize[0], eyeSize[1], 1)  # 2 eyes per image
XEyesTest = XEyesTest.reshape(-1, 2, eyeSize[0], eyeSize[1], 1)
XMouthTrain = XMouthTrain.reshape(-1, mouthSize[0], mouthSize[1], 1)
XMouthTest = XMouthTest.reshape(-1, mouthSize[0], mouthSize[1], 1)

# Define the CNN Model with Three Input Streams
def build_model():
    # Face Branch
    input_face = Input(shape=(128, 128, 1))
    x = Conv2D(32, (3, 3), activation='relu')(input_face)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    face_output = Dense(128, activation='relu')(x)

    # Eyes Branch (Processing Each Eye Separately)
    input_eyes = Input(shape=(2, 32, 32, 1))  # Two eyes per image
    
    # Reshape (Split into two separate (32,32,1) inputs)
    left_eye = Reshape((32, 32, 1))(input_eyes[:, 0, :, :, :])  
    right_eye = Reshape((32, 32, 1))(input_eyes[:, 1, :, :, :])

    # Apply Conv2D to each eye separately
    left_eye_features = Conv2D(16, (3, 3), activation='relu')(left_eye)
    left_eye_features = MaxPooling2D((2, 2))(left_eye_features)
    left_eye_features = Flatten()(left_eye_features)

    right_eye_features = Conv2D(16, (3, 3), activation='relu')(right_eye)
    right_eye_features = MaxPooling2D((2, 2))(right_eye_features)
    right_eye_features = Flatten()(right_eye_features)

    # Merge both eyes
    eyes_output = Concatenate()([left_eye_features, right_eye_features])
    eyes_output = Dense(64, activation='relu')(eyes_output)

    # Mouth Branch
    input_mouth = Input(shape=(64, 32, 1))
    z = Conv2D(16, (3, 3), activation='relu')(input_mouth)
    z = MaxPooling2D((2, 2))(z)
    z = Flatten()(z)
    mouth_output = Dense(64, activation='relu')(z)

    # Combine Face, Eyes & Mouth Features
    combined = Concatenate()([face_output, eyes_output, mouth_output])
    final = Dense(128, activation='relu')(combined)
    final = Dropout(0.3)(final)
    final = Dense(5, activation='softmax')(final)

    # Create Model
    model = Model(inputs=[input_face, input_eyes, input_mouth], outputs=final)
    return model

# Compile Model
model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit([XFaceTrain, XEyesTrain, XMouthTrain], yTrain, epochs=50, batch_size=64, validation_data=([XFaceTest, XEyesTest, XMouthTest], yTest))

# Save Model
model.save("face_emotion_model.keras")
print("Model training complete, saved as 'face_emotion_model_tf'")

# Evaluate Model
loss, accuracy = model.evaluate([XFaceTest, XEyesTest, XMouthTest], yTest)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
