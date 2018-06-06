import sys
import os
from PIL import Image, ImageDraw
import cv2
import numpy as np
import uuid
from matplotlib import pyplot as plt

def main():
	directory = os.fsencode(sys.argv[1])
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if filename.endswith(".bmp"): 
			path = os.path.join(directory.decode('utf-8'), filename)
			filename,_ = os.path.splitext(path)
			filename = os.path.basename(filename)
			im = cv2.imread(path,1)
			processFile(im, filename)
			continue
		else:
			continue

def processFile(im,filename):	
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	
	# Blur image to blur out details
	blur = cv2.GaussianBlur(gray,(9,9),0)

	#Use Morphological Closing and thresholding to get shape of object
	kernel = np.ones((5,5),np.uint8)
	opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	_, th = cv2.threshold(opening,127,255,0)
	
	kernel = np.ones((9,9),np.uint8)
	th = cv2.erode(th, kernel, iterations = 1)

	# Get minimal object bounding box
	im2,contours,hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	# Remove largest area because it ise the whole phone
	largest_area = max(contours, key=cv2.contourArea)	
	
	script_path=os.getcwd()
	os.makedirs(filename)
	os.chdir(filename)
	
	for cnt in contours:
		if np.array_equal(largest_area, cnt):
			continue
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
	
		# Rotate image to minimize background
		angle = rect[2]
		rows,cols = im.shape[0], im.shape[1]
		M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
		rot = cv2.warpAffine(im,M,(cols,rows))
		pts = np.int0(cv2.transform(np.array([box]), M))[0]	
		pts[pts < 0] = 0

		# Crop and save image
		crop = rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]
		if crop.size != 0:
			cv2.imwrite(str(uuid.uuid4()) + '.bmp', crop)
	os.chdir(script_path)
	
if __name__ == "__main__":
	main()

	
