import cv2
import os
import numpy as np



# Function to load the images and labels from the 'data/images' directory
def load_images_and_labels(images_dir):
    images = []
    labels = []
    for filename in os.listdir(images_dir):
        label = filename.split('_')[0]  # Extract the label from the filename
        image_path = os.path.join(images_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images.append(image)
        labels.append(label)
    return images, labels

##Function to preprocess image
def load_and_process_data(images_dir):
    images, labels = load_images_and_labels(images_dir)

    # Convert the images and labels to NumPy arrays
    faces_data = np.asarray(images)
    labels_data = np.asarray(labels)

    # Create a dictionary to map the unique labels to integer labels
    label_to_id = {label: idx for idx, label in enumerate(np.unique(labels_data))}

    # Convert the labels to integer labels based on the mapping
    labels_data_ids = np.array([label_to_id[label] for label in labels_data], dtype=np.int32)

    return faces_data, labels_data_ids

# Function to get the label from the label_id
def get_label_from_id(label_id, label_to_id):
    return [label for label, id in label_to_id.items() if id == label_id][0]