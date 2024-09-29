import cv2
from deepface import DeepFace
#DeepFace is a lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for python.

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#CascadeClassifier object in OpenCV is a class that applies the algorithm for detecting objects based on the data in the XML file. It does the actual work of processing the image, searching for patterns that match the features described in the XML, and returning regions where faces are detected.
#haarcascade_frontalface_default.xml file is a pre-trained model containing all the parameters, thresholds, and Haar-like features for detecting faces. It's essentially a blueprint or database of face features.
#When we pass the XML file to CascadeClassifier, OpenCV loads the data about face features from the XML. making it capable of detection 
#Once the classifier is linked with the trained data (from the XML), we can apply it to any image or video frame by calling detectMultiScale()

#capturing video
cap = cv2.VideoCapture(0)

while True:
   
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)


    #This function scans the image (in grayscale) and tries to match sections of the image with the patterns described in the Haar cascade. If it finds a match, it returns the coordinates of the detected face.
    #grayscale for better efficiency. As gray scale is single channel therefore less computations of pixel values (contrast to rgb with 3 channels)
    #parameters:
        #scale factor - algorithm needs to "rescale" the image to find faces of varying sizes. At each scale, the algorithm shrinks the image slightly and looks for faces, then resizes it again, and repeats the process. lower value means more scales hence better accuracy
        #minNeighbours - It's a threshold for how confident the algorithm needs to be that a detected region is indeed a face. A higher value like 5 means that more rectangles (detections) must overlap for the algorithm to confirm it as a face, reducing false positives.
        #min size - algorithm will ignore any objects smaller than 30x30 pixels as potential faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))



    for (x, y, w, h) in faces:  
        #tuples for diff coordinates of a face
        # x-x coord, y-y coord, w-width, h-height
        #face roi - region of interest : portion of image containing the face
        
        #a slicing operation used to extract a rectangular area 
        #slicing syntax [start:stop]
        #face is detected (using the grayscale image), the corresponding color information is extracted from the rgb_frame to perform emotion analysis.
        face_roi = rgb_frame[y:y + h, x:x + w]
        
        
        
        #emotion analysis on the face ROI
        #third parameter-an additional face detection step to make sure the input image has a face. Setting this to False bypasses that since you’ve already detected the face using OpenCV.
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=True)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion'] #dominant emotion key? returned from dictionary

        # Draw rectangle around face and label with predicted emotion, in same frame and with coordinates, colour (rg) and thickness of rect line specified
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        #text appears of 'emotion', at specified coordinates, font, r=color, thickness
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
   
    
    cv2.imshow('Real-time Emotion Detection', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()