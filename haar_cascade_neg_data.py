import numpy as np
import cv2
import os

f = open('cascade_data/bg.txt', 'a')
pos_list = os.listdir('temp_data/forest')

for image in pos_list:
	if image.endswith(".jpeg"):
		file_name = 'temp_data/forest/' + image
		data = cv2.imread(file_name)
		data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
		data = cv2.resize(data, (100,100))
		cv2.imshow('data',data)

		directory = 'cascade_data/neg/' + image
		cv2.imwrite(directory, data)
		string = "neg/" + image + "\n"
		f.write(string)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cv2.destroyAllWindows()