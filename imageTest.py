import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

model_data = np.load("expression_detector_model.npz")
W1, b1, W2, b2 = model_data["W1"], model_data["b1"], model_data["W2"], model_data["b2"]
categories = ["angry", "happy", "neutral", "sad", "surprise"]

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def relu(z):
    return np.maximum(0, z)

def predict_expression(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not read image.")
        return
    
    img = cv2.resize(img, (48, 48)).flatten() / 255.0
    
    Z1 = np.dot(img, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    
    predicted_expression = categories[np.argmax(A2)]
    print(f"Predicted Expression: {predicted_expression}")
    
    img_display = cv2.imread(image_path)
    cv2.putText(img_display, predicted_expression, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Predicted Expression", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def browse_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        predict_expression(file_path)

if __name__ == "__main__":
    browse_image()
