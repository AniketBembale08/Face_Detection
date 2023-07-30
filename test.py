import cv2
import os
import numpy as np
from Data_preprocess import load_images_and_labels,get_label_from_id



# Function to recognize faces in real-time
def real_time_face_recognition(model_file):
    # Load the trained model
    trained_model = cv2.face.LBPHFaceRecognizer_create()
    trained_model.read(model_file)

    # Load the face images and labels and preprocess the data
    data_dir = 'data'
    images_dir = os.path.join(data_dir, 'images')
    faces_data, labels_data_ids = load_images_and_labels(images_dir)

    # Create a dictionary to map the unique labels to integer labels
    label_to_id = {label: idx for idx, label in enumerate(np.unique(labels_data_ids))}

    # Start capturing video from the webcam
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            # Perform prediction on the face region of interest
            label_id, confidence = trained_model.predict(face_roi)

            # Get the recognized label from the label_id
            recognized_label = get_label_from_id(label_id, label_to_id)

            # Display the recognized label and confidence on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text1 = f"Name: {recognized_label}"
            cv2.putText(frame, text1, (x, y - 10), font, font_scale, (0, 255, 0), font_thickness)
            text2 = f"Conf: {confidence:.2f}"
            cv2.putText(frame, text2, (x, y +180), font, font_scale, (0, 255, 0), font_thickness)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        cv2.imshow('Real-time Face Recognition', frame)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# File path of the trained model
model_file = 'data/model1.yml'
# Call the function to perform real-time face recognition using the trained model
real_time_face_recognition(model_file)
