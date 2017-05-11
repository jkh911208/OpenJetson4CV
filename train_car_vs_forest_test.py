import numpy as np
from googlenet import googlenet
# from googlenet import googlenet

WIDTH = 90
HEIGHT = 90
LR = 1e-3
epoch = 10

model = googlenet(WIDTH,HEIGHT,LR)

# googlenet = googlenet(width,height,1,LR)

for i in range(epoch):
	train_data = np.load('car_vs_forest.npy')

	train = train_data

	X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
	Y = [i[1] for i in train]

	model.fit(X,Y, n_epoch=1, validation_set=0.1,shuffle=True,snapshot_step=500, show_metric=True, run_id="googlenet")

	model.save("googlenet.model")