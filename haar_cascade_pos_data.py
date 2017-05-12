import numpy as np
import cv2
import os

f = open('cascade_data/pos.txt', 'a')
pos_list = os.listdir('temp_data/car')

for image in pos_list:
	if image.endswith(".jpeg"):
		file_name = 'temp_data/car/' + image
		data = cv2.imread(file_name)
		data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
		data = cv2.resize(data, (50,50))
		cv2.imshow('data',data)
		#cv2.imshow('data',data)
		directory = 'cascade_data/pos/' + image
		cv2.imwrite(directory, data)
		string = "pos/" + image + " 1 0 0 50 50\n"
		f.write(string)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cv2.destroyAllWindows()