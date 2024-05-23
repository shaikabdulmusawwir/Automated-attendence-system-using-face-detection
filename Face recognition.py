import cv2

# Local Binary Patterns Histograms
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')  # Load trained model
cascadePath = "haarcascade_frontalface_default.xml"
# initializing haar cascade for object detection approach
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX  # Denotes the font type

id = 4

names = ['kiran', 'pavan', 'Bkiran', 'abdul']

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

# flag = true

while True:
    ret, img = cam.read()  # Reads frame using above created object
    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # The function converts an input image from one form to other 

    faces = faceCascade.detectMultiScale(
        converted_image,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
        )
    for(x,y,w,h) in faces:
        
        cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),2) # TO draw a rectangle on any image
        id, accuracy = recognizer.predict(converted_image[y:y+h,x:x+w])
        # Check if accuracy is less then 100 ==> "0" is perfect match
        if (accuracy < 100):
            id = names[id]
            accuracy = "  {0}%".format(round(100 - accuracy))

        else:
            id = "unknown"
            accuracy = "  {0}%".format(round(100 - accuracy))
            
        cv2.putText(img ,str(id), (x+5,y-5), font, 1, (255, 255,255), 2)
        cv2.putText(img ,str(accuracy), (x+5,y+h-5), font, 1, (255, 255,255), 1)

    cv2.imshow('camera',img)
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do  abit of cleanup
print("Thanks for using this program, have a good day.")
cam.release()
cv2.destroyAllWindows()
