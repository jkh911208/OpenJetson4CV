import numpy as np
import cv2
import os

file_list = os.listdir('temp_data')
#print(file_list)

for file in file_list:
  if file.endswith(".npy"):
  	print (file)
  	file_name = "temp_data/" + file
  	data = np.load(file_name)
  	print(data)
  	# for data in file:
  	# 	print(data[0])
    

# raw_data = np.load("1494431577.559893.npy")
# print(raw_data[0][0].shape)

# for data in raw_data:
#     front_frame = data[0]
#     front_frame = cv2.flip(front_frame,0)
#     front_frame = cv2.flip(front_frame,1)

#     cv2.imshow('front_frame',front_frame)


#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# #front_frame.release()
# cv2.destroyAllWindows()