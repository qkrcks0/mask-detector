from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import sys

def detect(target):

    # blob object
    blob = cv2.dnn.blobFromImage(target, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
    face_detector.setInput(blob)
    out = face_detector.forward()

    # detect faces
    detected = out[0, 0, :, :]
    (h,w) = target.shape[:2]

    for i in range(detected.shape[0]):

        confidence = detected[i, 2]

        if confidence < 0.3:
            break

        x1 = int(detected[i, 3] * w)
        y1 = int(detected[i, 4] * h)
        x2 = int(detected[i, 5] * w)
        y2 = int(detected[i, 6] * h)

        face_img = target[y1:y2, x1:x2].copy()

        face_input = cv2.resize(face_img, dsize=(224, 224))
        face_input = preprocess_input(face_input)
        face_input = np.expand_dims(face_input, axis=0)
        
        mask, nomask = mask_detect_model.predict(face_input).squeeze()

        if mask > nomask:
            labels = f"Mask {round(mask*100)}%"
            # draw rectangle around the face object
            cv2.rectangle(target, (x1, y1), (x2, y2), (0, 255, 0))
            cv2.putText(target, labels, (x1, y1-1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            labels = f"No Mask {round(nomask*100)}%"
            # draw rectangle around the face object
            cv2.rectangle(target, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.putText(target, labels, (x1, y1-1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return target

def camera():
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
    
        frame = detect(frame)
    
        cv2.imshow("frame", frame)

        if cv2.waitKey(round(1000/fps)) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def images():

    for i, img_path in enumerate(imgs_list):

        img = cv2.imread(img_path)
        
        if img is None:
            sys.exit()

        img = detect(img)

        cv2.imwrite(f'result/result{i}.jpg', img)

        cv2.imshow("frame", img)
        if cv2.waitKey() == ord(" "):
            continue

# face detect model
face_detector = cv2.dnn.readNetFromCaffe('models/deploy.prototxt',\
                                         'models/res10_300x300_ssd_iter_140000_fp16.caffemodel')

# mask detect model
mask_detect_model = load_model("models\mask_detector.model")

# image list
imgs_list = ["imgs/978_1032_3920.jpg", "imgs/200319_142174_4346.jpg", \
             "imgs/103891218.2.jpg", "imgs/2020082401257_0.jpg", \
             "imgs/art_16049744339779_c1abc8.jpg", \
             "imgs/art_16051493926854_0f7653.jpg", "imgs/images.jpg"]

# camera()
images()



