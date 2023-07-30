# Face_Detection

## ⭐️Introduction
This is a face detection project that uses Haarcascade and Computer Vision to collect user image data, preprocess the data, train a model using LBPHFaceRecognizer, and finally perform face detection using the trained model.

## ⭐️Overview
This is a face detection project that utilizes computer vision and machine learning techniques to detect faces in images. The project consists of several modules for data collection, data preprocessing, model training, and face detection.

## ⭐️Project Structure

    face-detection/
    |-- data/
        |-- images
    |-- models/
        |-- trained.yaml
        |-- haarcascade_frontalface_default.xml
    |-- collect_data.py
    |-- data_preprocess.py
    |-- train.py
    |-- test.py


    

## ⭐️Dependencies

OpenCV: Library for computer vision tasks, including face detection and image processing.
NumPy: Library for numerical computations in Python.
You can find the complete list of dependencies with their versions in the requirements.txt file.

## ⭐️How to Use
Instructions

Clone the repository: git clone https://github.com/AniketBembale08/AI-Disease-Diagnosis.git

Install the required dependencies: 

pip install -r requirements.txt

Run *collect_data.py* to collect user image data. It will prompt you to enter the user's name and capture images using your webcam.

Run *data_preprocess.py* to process the collected data and create labels for each user.

Run *train.py* to train the model using LBPHFaceRecognizer.

After training, the trained model will be saved as *model.yaml*.

Run *test.py* to perform face detection using the trained model. The program will use your webcam to detect faces in real-time.

## ⭐️File Descriptions
*collect_data.py:* Collects user image data using Haarcascade and Computer Vision.

*data_preprocess.py:* Processes the collected data and creates labels for each user.

*train.py:* Trains the face detection model using LBPHFaceRecognizer.

*test.py:* Performs face detection using the trained model.

## ⭐️How the Model Works
The face detection model is trained using LBPHFaceRecognizer, a Local Binary Patterns Histogram-based face recognition algorithm. It learns to recognize facial features and patterns from the collected data and can then detect faces in real-time using the trained model.

## ⭐️Credits
*Haarcascade:* OpenCV's implementation of Haar feature-based cascade classifiers.

*LBPHFaceRecognizer:* The Local Binary Patterns Histogram-based face recognition algorithm from OpenCV.


Feel free to contribute to this project by submitting pull requests or opening issues. Happy face detection! 😄
















