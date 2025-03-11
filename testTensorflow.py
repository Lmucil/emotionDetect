import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
                X.append(img)
                y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)

XTrain, yTrain = load_images_from_folder(trainDatasetPath)
XTest, yTest = load_images_from_folder(testDatasetPath)

yTrain = to_categorical(yTrain, numClasses)
yTest = to_categorical(yTest, numClasses)

XTrain = XTrain.reshape(-1, imageSize[0], imageSize[1], 1)
XTest = XTest.reshape(-1, imageSize[0], imageSize[1], 1)

# Data augmentation 
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(XTrain)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(imageSize[0], imageSize[1], 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),  # Helps prevent overfitting

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(numClasses, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(datagen.flow(XTrain, yTrain, batch_size=64),
          epochs=50, 
          validation_data=(XTest, yTest))

model.save("face_detector_model.h5")
print("Model training complete, saved as 'face_detector_model.h5'")
loss, accuracy = model.evaluate(XTest, yTest)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
