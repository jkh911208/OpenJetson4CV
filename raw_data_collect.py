import numpy as np
import cv2
import time
import datetime

cap = cv2.VideoCapture(0)

cap.set(3,245) # set height
cap.set(4,300) # set width

ret, frame = cap.read()
height, width = frame.shape[0], frame.shape[1]
print(frame.shape)
raw_data = []
while(ret):
    _, frame = cap.read()
    raw_data.append([frame])
    cv2.imshow('frame',frame)

    if len(raw_data) % 100 == 0:
        file_name = str(datetime.datetime.now()) + ".npy"
        print(len(raw_data))
        np.save(file_name,raw_data)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()