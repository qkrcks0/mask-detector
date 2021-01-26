from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import sys
# import dlib

# face detect model
# face_detector = dlib.get_frontal_face_detector()
face_detector = cv2.dnn.readNet('models/deploy.prototxt', \
                                'models/res10_300x300_ssd_iter_140000.caffemodel')

# mask detect model
mask_detect_model = load_model("models\mask_detector.model")

# open camera
cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)

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
        
        mask, nomask = mask_detect_model.predict(face_input).squeeze()

        # draw rectangle around the face object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))
        if mask > nomask:
            labels = f"Mask {round(mask*100)}%"
            cv2.putText(frame, labels, (x1, y1-1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            labels = f"No Mask {round(nomask*100)}%"
            cv2.putText(frame, labels, (x1, y1-1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)

    if cv2.waitKey(round(1000/fps)) == 27:
        break

cap.release()
cv2.destroyAllWindows()

