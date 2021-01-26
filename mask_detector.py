from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import sys
import dlib

# face detect model
face_detector = dlib.get_frontal_face_detector()

# mask detect model
mask_detect_model = load_model("models\mask_detector.model")

# open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # detect faces
    faces = face_detector(frame)

    for face in faces:

        x1,y1,x2,y2 = face.left(), face.top(), face.right(), face.bottom()

        face_img = frame[y1:y2, x1:x2].copy()

        face_input = cv2.resize(face_img, dsize=(224, 224))
        face_input = preprocess_input(face_input)
        face_input = np.expand_dims(face_input, axis=0)
        # print(face_input.shape)
        
        mask, nomask = mask_detect_model.predict(face_input).squeeze()
        print(mask, nomask)

        # draw rectangle around the face object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.imshow("frame", frame)

    if cv2.waitKey() == ord(" "):
        continue
    
cv2.destroyAllWindows()

