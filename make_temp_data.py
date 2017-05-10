import numpy as np
import cv2
import time

raw_data = []
val1 = 0
val2 = 10000000
val3 = 100000000000000
count = 0

while(True):
    raw_data.append([val1,val2,val3])
    val1 += 1
    val2 += 1
    val3 += 1

    if len(raw_data) % 100 == 0:
        file_name = "temp_data/" + str(time.time()) + ".npy"
        np.save(file_name,raw_data)
        row_data = []
        count += 1

    if count == 5000:
    	break;

cv2.destroyAllWindows()