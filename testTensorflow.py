import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Load the FER-2013 dataset manually
def load_fer2013(dataset_path):
    images, labels = [], []
    class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    with open(dataset_path, 'r') as file:
        next(file)  # Skip header
        for line in file:
            pixels, emotion = line.strip().split(',')[1:]
            image = np.array(pixels.split(), dtype=np.uint8).reshape(48, 48)  # FER-2013 images are 48x48
            images.append(image)
            labels.append(int(emotion))

    images = np.array(images) / 255.0  # Normalize images
    labels = np.array(labels)
    return images, labels, class_names

# Update path to your dataset CSV file
dataset_path = "fer2013.csv"
images, labels, class_names = load_fer2013(dataset_path)

# Split dataset into training and testing
num_samples = len(images)
split_index = int(0.8 * num_samples)  # 80% training, 20% testing

train_images, test_images = images[:split_index], images[split_index:]
train_labels, test_labels = labels[:split_index], labels[split_index:]

# Reshape images for CNN input
train_images = train_images.reshape(-1, 48, 48, 1)
test_images = test_images.reshape(-1, 48, 48, 1)

# Display some images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i].reshape(48, 48), cmap="gray")
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')  # 7 classes in FER-2013
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc:.2f}")
