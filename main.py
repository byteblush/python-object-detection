import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = []
classFile = 'config/class_names.txt'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'config/detection_model_config.txt'
weightsPath = 'config/detection_model_weights.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

COLORS = np.random.uniform(0, 255, size=(len(classNames), 3))

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            idx = (classId - 1)
            detected_object = classNames[idx]
            color = COLORS[idx]

            cv2.rectangle(img, box, color, thickness=5)
            cv2.putText(img, detected_object, (box[0] + 10, box[1] + 30),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2)

    cv2.imshow('output', img)
    cv2.waitKey(10)
