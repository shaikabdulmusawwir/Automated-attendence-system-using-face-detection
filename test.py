import cv2
import pyttsx3
from PIL import Image

functions = dir(pyttsx3)
for f in functions:
    print(f)
print(cv2.__version__)
