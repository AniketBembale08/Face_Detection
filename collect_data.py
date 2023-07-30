import cv2
import os

def collect_face_data(data_dir='data', max_faces=100, face_size=(50, 50)):
    images_dir = os.path.join(data_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    faces_data = []
    i = 0
    name = input("Enter Your Name: ")

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, face_size)
            if len(faces_data) <= max_faces and i % 10 == 0:
                faces_data.append(resized_img)
                # Save the face image to a file
                image_path = os.path.join(images_dir, f'{name}_{len(faces_data)}.png')
                cv2.imwrite(image_path, resized_img)
            i = i + 1
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == ord('q') or len(faces_data) == max_faces:
            break
    video.release()
    cv2.destroyAllWindows()

    print("Data Added Successfully!!!!!!!!!!!!!!!!!!")

# Call the function to collect face data
collect_face_data()
