from cv2 import cv2
import numpy as np 

thres_val = 0.5
nms_threshold = 0.2

# Load the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

# Load class files
class_names = []
class_file = 'coco.names'
with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# print(f'Class Names are:-\n{class_names}')

config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weight_path = 'frozen_inference_graph.pb'

# This is the backend work
net = cv2.dnn_DetectionModel(weight_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    class_ids, confs, bounding_box = net.detect(img, confThreshold=thres_val)
    bounding_box = list(bounding_box)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    # Used NMS for drop repeating boxes on the object
    indices = cv2.dnn.NMSBoxes(bounding_box, confs, thres_val, nms_threshold)
    for i in indices:
        i = i[0]
        box = bounding_box[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)
        cv2.putText(img, class_names[class_ids[i][0]-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

    # # print(class_ids, bounding_box)
    # if len(class_ids) != 0:
    #     for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bounding_box):
    #         cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

    #         # Show name of the image/object
    #         cv2.putText(img, class_names[class_id - 1].upper(), (box[0]+10, box[1]+30), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

    #         # Show how the accurate is our prediction
    #         cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

    cv2.imshow('Output', img)
    # if cv2.waitKey(1):
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()