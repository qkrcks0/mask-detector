from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import sys
# import dlib

# face detect model
# face_detector = dlib.get_frontal_face_detector()
face_detector = cv2.dnn.readNetFromCaffe('models/deploy.prototxt',\
                                         'models/res10_300x300_ssd_iter_140000_fp16.caffemodel')

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
    
    # blob object
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
    face_detector.setInput(blob)
    out = face_detector.forward()

    # detect faces
    detect = out[0, 0, :, :]
    (h,w) = frame.shape[:2]

    for i in range(detect.shape[0]):

        confidence = detect[i, 2]

        if confidence < 0.5:
            break

        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

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

