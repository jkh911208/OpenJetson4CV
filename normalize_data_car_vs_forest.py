import numpy as np
import cv2
import os
import time
from random import shuffle 

car_list = os.listdir('temp_data/car')
forest_list = os.listdir('temp_data/forest')
#print(file_list)
car = []
forest = []
car_one_hot = [1,0]
forest_one_hot = [0,1]

for file in car_list:
	if file.endswith(".jpeg"):
	  	file_name = "temp_data/car/" + file
	  	# print(file)
	  	image = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
	  	image = cv2.resize(image,(90,90))
	  	# print(square_image.shape)
	  	# cv2.imshow('square_image',image)
	  	car.append([image, car_one_hot])

	  	if cv2.waitKey(1) & 0xFF == ord('q'):
	  		break

for file in forest_list:
	if file.endswith(".jpeg"):
	  	file_name = "temp_data/forest/" + file
	  	# print(file)
	  	image = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
	  	image = cv2.resize(image,(90,90))
	  	# print(image.shape)
	  	# cv2.imshow('square_image',image)
	  	forest.append([image, forest_one_hot])

	  	if cv2.waitKey(1) & 0xFF == ord('q'):
	  		break

final_data = car + forest

shuffle(final_data)
np.save('car_vs_forest.npy', final_data)

# data = []
# # label = []
# for i in final_data:
# 	# reshape_data = i[0].reshape(-1,90,90,1)
# 	# data.append([reshape_data])
# 	# label.append([i[1]])
# 	print(i[1])


# np.save('car_vs_forest_data.npy', data)
# np.save('car_vs_forest_label.npy', label)

cv2.destroyAllWindows()