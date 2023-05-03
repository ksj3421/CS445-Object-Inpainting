# Note, this requires cloning the facenet_pytorch repo
from facenet_pytorch import MTCNN
import numpy as np
import cv2
import matplotlib.pyplot as plt

device = 'cpu'
def detect_face(img):
    mtcnn = MTCNN(keep_all=True, device=device)
    return mtcnn.detect(img)

def mask_face(img, x1, x2, y1, y2):
    img[y1:y2,x1:x2] = np.random.rand(y2-y1, x2-x1, 3) * 255
    return img

def mask_detected_faces(input_face_path='data_faces/img_align_celeba/000100.jpg'):
    frame = cv2.cvtColor(cv2.imread(input_face_path), cv2.COLOR_BGR2RGB)
    boxes, confidence = detect_face(frame)

    masked = frame.copy()

    for i, detected in enumerate(boxes):
        if confidence[i] < .5:
            continue

        [x1, y1, x2, y2] = detected
        
        # adjust mask smaller so face replacement has more to work with
        x_adj = abs(x2-x1) / 8
        y_adj = abs(y2-y1) / 8
        x1 = int(x1 + x_adj)
        x2 = int(x2 - x_adj)
        y1 = int(y1 + y_adj)
        y2 = int(y2 - y_adj)
        
        masked = mask_face(masked, x1, x2, y1, y2)

    plt.imshow(masked)
    plt.show()
    return masked

mask_detected_faces()
