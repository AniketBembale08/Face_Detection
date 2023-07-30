import cv2
import os
import numpy as np
from Data_preprocess import load_and_process_data,load_images_and_labels,get_label_from_id


data_dir = 'data'
images_dir = os.path.join(data_dir, 'images')


faces_data, labels_data_ids = load_and_process_data(images_dir)


# Create the face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the model using the face images and labels
recognizer.train(faces_data, labels_data_ids)

# Save the trained model to a file
model_file = os.path.join(data_dir, 'model1.yml')
recognizer.save(model_file)

print("***********************************************")
print("Model Trained Successfully!!!!!!!!!!!!!!!!!!")
