import numpy as np
import cv2
import os
from random import shuffle 
from alexnet import alexnet
from googlenet import googlenet

width = 100
height = 100
LR = 1e-3
epoch = 3

alexnet = alexnet(width,height,1,LR)
googlenet = googlenet(width,height,1,LR)

loaded_data = np.load('car_vs_forest.npy')

train = loaded_data[:-100]
test = loaded_data[-100:]

for i in train:
	train_x = i[0]
	train_x = train_x.reshape(-1, width,height,1)
	train_y = i[1]

# train_x = np.array(i[0] for i in train).reshape(-1,width,height,1)
# train_y = np.array(i[1] for i in train)


for i in test:
	test_x = i[0]
	test_x = test_x.reshape(-1, width,height,1)
	test_y = i[1]

# test_x = np.array(i[0] for i in test).reshape(-1,width,height,1)
# test_x = np.array(i[1] for i in test)


alexnet.fit(train_x,train_y, n_epoch = epoch, validation_set =0.1, shuffle=True,
	snapshot_step=500, show_metric=True, run_id="alexnet")
alexnet.save("alexnet.model")

googlenet.fit(train_x,train_y, n_epoch = epoch, validation_set =0.1,shuffle=True,
	snapshot_step=500, show_metric=True, run_id="googlenet")
googlenet.save("googlenet.model")