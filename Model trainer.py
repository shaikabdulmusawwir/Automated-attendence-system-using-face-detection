import cv2
import numpy as np
from PIL import Image
import os
import glob

path = 'samples'  # path for samples taken already

# Local Binary Patterns Histograms
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def Images_And_Labels(path):  # Function to fetch the images and labels

    # imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    imagePaths = [f for f in glob.glob(path + '*.jpg')]
    faceSamples = []
    ids = []
    
    for imagePath in imagePaths:  # To iterate particular image path
        gray_img = Image.open(imagePath).convert('L')  # To convert to greyscale
        img_arr = np.array(gray_img, 'uint8')  # Creating an Array
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_arr)
        for (x, y, w, h) in faces:
            faceSamples.append(img_arr[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids
print("Training faces. It Will take a few Seconds. please wait.....!")

faces, ids = Images_And_Labels(path)
recognizer.train(faces, np.array(ids))
# Save the Trained model as Trainer.yml
recognizer.write('trainer/trainer.yml')

print("Model trained, Now we can recognize your face.") 