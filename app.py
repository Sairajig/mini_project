import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# ============================
# 1. Data Preparation
# ============================
def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    class_mapping = {
        "MildDemented": 0,
        "ModerateDemented": 1,
        "NonDemented": 2,
        "VeryMildDemented": 3
    }

    for class_name, label in class_mapping.items():
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (256, 256))  # Resize to match model input
            image = img_to_array(image) / 255.0  # Normalize
            images.append(image)
            labels.append(label)

    return np.array(images), tf.keras.utils.to_categorical(labels, num_classes=4)

# ============================
# 2. Classification Model
# ============================
def build_classification_model():
    base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)  # Four classes
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ============================
# 3. Flask Integration
# ============================
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = f"uploads/{file.filename}"
    file.save(file_path)

    # Load and preprocess the image
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image, (256, 256))
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Perform Classification
    classification_model = load_model('path/to/classification_model.h5')
    classification_result = classification_model.predict(image_array)
    class_label = np.argmax(classification_result)
    stage = {
        0: 'Non-Demented',
        1: 'Very Mild Demented',
        2: 'Mild Demented',
        3: 'Moderate Demented'
    }
    confidence = classification_result[0][class_label]

    # Generate Report
    report = f"""MRI Analysis Report:\nStage: {stage[class_label]}\nConfidence: {confidence:.2f}\n"""
    return jsonify({'stage': stage[class_label], 'confidence': confidence, 'report': report})

if __name__ == '__main__':
    # Prepare and train the model if not already trained
    data_dir = "C:/Users/rajig/.vscode/cli/vs code/bin/mini_project/Alzheimer_MRI_4_classes_dataset"
    images, labels = load_and_preprocess_data(data_dir)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Build and train the classification model
    model = build_classification_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    model.save("path/to/classification_model.h5")

    # Run the Flask app
    app.run(debug=True)
