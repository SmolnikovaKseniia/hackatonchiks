import cv2
import numpy as np
import os

# Minimum size threshold for considering images, in bytes
MIN_SIZE = 1024 * 15

# Folder path containing images
FOLDER_PATH = "C:\\Users\\nikit\\Downloads\\wiki_crop"


def get_image_size(filename: str) -> int:
    """
    Function to get image size
    -----------------
    -----------------
    Args:
        filename (str): name of the file
    -----------------
    Returns:
        int: the size in the byte
    """
    sz = os.stat(filename)
    return sz.st_size


def get_image_pathes(folder_path: str) -> list[str]:
    """
    Function to get pathes of images
    -----------------
    -----------------
    Args:
        folder_path (str): folder path of the image
    -----------------
    Returns:
        list[str]: list of images' pathes 
    """
    image_pathes = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_pathes.append(os.path.join(root, file)[:-4])

    return image_pathes


def get_face(image_path: str) -> np.ndarray:
    """
    Extracts the face from an image using Haar Cascade classifier
    -----------------
    -----------------
    Args:
        image_path (str): The path to the image file
    -----------------
    Returns:
        np.ndarray: The grayscale image of the detected face
    -----------------
    Note:
      - 1) Ensure that OpenCV (cv2) library is installed
      - 2) The MIN_SIZE variable should be predefined and accessible in the current scope

    """
    image = cv2.imread(f'{image_path}.jpg')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    # If no face is detected or image size is smaller than the threshold, return empty array
    if len(faces) == 0 or get_image_size(f'{image_path}.jpg') < MIN_SIZE:
        return []

    # Iterate over detected faces to find the frontal face
    for (x, y, w, h) in faces:
        # Check if the face is frontal
        if w > 0.8 * h:  # Assuming frontal faces are wider than taller
            # Extract the detected face
            face = image[y:y+h, x:x+w]
            face_bw = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            return face_bw

    return []


# Get paths of all image files in the folder
IMAGE_PATHES = get_image_pathes(FOLDER_PATH)

# Initialize a counter
i = 0

for name in IMAGE_PATHES:
    result = get_face(name)

    # If a face is detected, save it to the result folder
    if len(result) != 0:
        # Resize the detected face to 256x256
        resized_face = cv2.resize(result, (256, 256))
        cv2.imwrite(f'result_folder/{i + 1}.jpg', resized_face)
        i += 1

    # Break loop after processing 9000 images
    if i == 9000:
        break