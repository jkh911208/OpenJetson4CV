import numpy as np
import cv2


raw_data = np.load("1494431577.559893.npy")
print(raw_data[0][0].shape)

for data in raw_data:
    front_frame = data[0]
    front_frame = cv2.flip(front_frame,0)
    front_frame = cv2.flip(front_frame,1)

    cv2.imshow('front_frame',front_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()