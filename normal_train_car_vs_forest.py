import numpy as np
import cv2
import os
import time
from random import shuffle 
from tqdm import tqdm

car_list = os.listdir('temp_data/car')
forest_list = os.listdir('temp_data/forest')
#print(file_list)
car = []
forest = []
car_one_hot = [1,0]
forest_one_hot = [0,1]

for file in tqdm(car_list):
	if file.endswith(".jpeg"):
	  	file_name = "temp_data/car/" + file
	  	image = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
	  	image = cv2.resize(image,(90,90))
	  	car.append([image, car_one_hot])

	  	if cv2.waitKey(1) & 0xFF == ord('q'):
	  		break

for file in tqdm(forest_list):
	if file.endswith(".jpeg"):
	  	file_name = "temp_data/forest/" + file
	  	image = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
	  	image = cv2.resize(image,(90,90))
	  	forest.append([image, forest_one_hot])

	  	if cv2.waitKey(1) & 0xFF == ord('q'):
	  		break

final_data = car + forest

shuffle(final_data)
# np.save('car_vs_forest.npy', final_data)

'''
Make the useable data from the image
'''
# start traing

from googlenet import googlenet

WIDTH = 90
HEIGHT = 90
LR = 1e-3
epoch = 10

model = googlenet(WIDTH,HEIGHT,LR)

for i in range(epoch):
	# train_data = np.load('car_vs_forest.npy')
	train_data = final_data

	train = train_data

	X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
	Y = [i[1] for i in train]

	model.fit(X,Y, n_epoch=1, validation_set=0.1,shuffle=True,snapshot_step=500, show_metric=True, run_id="googlenet")

	model.save("googlenet.model")


# data = []
# # label = []
# for i in final_data:
# 	# reshape_data = i[0].reshape(-1,90,90,1)
# 	# data.append([reshape_data])
# 	# label.append([i[1]])
# 	print(i[1])


# np.save('car_vs_forest_data.npy', data)
# np.save('car_vs_forest_label.npy', label)