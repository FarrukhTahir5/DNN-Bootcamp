# import os
# import face_recognition
# print(os.listdir('./Projects/Smart Attendance System/dataset'))
# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = face_recognition.load_image_file(os.path.join(folder, filename))
#         images.append(img)
#     return images

# def encode_faces(dataset_path):
#     known_face_encodings = []
#     known_face_names = []

#     for person_name in os.listdir(dataset_path):
#         person_folder = os.path.join(dataset_path, person_name)
#         person_images = load_images_from_folder(person_folder)
        
#         for image in person_images:
#             face_encodings = face_recognition.face_encodings(image)
#             if face_encodings:
#                 known_face_encodings.append(face_encodings[0])
#                 known_face_names.append(person_name)
    
#     return known_face_encodings, known_face_names

# dataset_path = "./Projects/Smart Attendance System/dataset"
# known_face_encodings, known_face_names = encode_faces(dataset_path)

import os
import face_recognition
import numpy as np
import cv2

# Directory containing training images
train_dir = './Projects/Smart Attendance System/dataset'

# Load training images and their labels
known_face_encodings = []
known_face_labels = []

for person_name in os.listdir(train_dir):
    person_dir = os.path.join(train_dir, person_name)
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_labels.append(person_name)

# Function to predict the label of a given image
def predict_label(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if not face_encodings:
        print("No face found in the image.")
        return

    face_encoding = face_encodings[0]
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        label = known_face_labels[best_match_index]
        print(f"Predicted label: {label}")
    else:
        print("No match found.")

# Train the model (this step is just loading the encodings)
print("Training complete.")

# Predict the label for a new image
test_image_path = './Projects/Smart Attendance System/Aaron_Eckhart_0001.jpg'
predict_label(test_image_path)
