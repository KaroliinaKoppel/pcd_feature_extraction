import sys
import os
import numpy as np
import cv2 
from skimage import io, morphology, img_as_bool, segmentation, img_as_ubyte
from scipy import ndimage as ndi

def main():
	img = readImage(sys.argv[1])
	canny = findCannyEdges(img)
	result = findFaults(canny)
	filename, extension = os.path.splitext(sys.argv[1])
	cv2.imwrite(filename + '_result'+extension, result)
	cv2.imshow('img',result)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def readImage(fullname):
	filename, extension = os.path.splitext(fullname)
	if (extension != '.bmp'):
		print('File type must be .bmp')
		return -1
	img = cv2.imread(fullname,1)
	return img

def findCannyEdges(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
	canny = cv2.Canny(gray,100,200)
	return np.uint8(canny)

def findFaults(img):
	camera_cascade = cv2.CascadeClassifier('camera_cascade.xml')
	fingerprint_sensor_cascade = cv2.CascadeClassifier('fingerprint_sensor_cascade.xml')
	speaker_cascade = cv2.CascadeClassifier('speaker_cascade.xml')
	
	kernel = np.ones((5,5),np.uint8)
	img = cv2.dilate(img,kernel,iterations = 1)
	img = img_as_ubyte(fillInContours(img))
	#img = cv2.dilate(img,kernel,iterations = 1)
	_,contours,hierarchy = cv2.findContours(img, 1, 2)
	basename = sys.argv[1]
	filename, extension = os.path.splitext(sys.argv[1])
	img = readImage(basename)
	img_color = readImage(basename)
	count = 0
	rows,cols = img.shape[0], img.shape[1]
	for cnt,hier in zip(contours,hierarchy[0]):
		area = cv2.contourArea(cnt)
		if (area > 15):
			rect = cv2.minAreaRect(cnt)
			box = cv2.boxPoints(rect)
			rect_box = np.int0(box)
			angle = rect[2]
			M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
			box = np.int0(cv2.transform(np.array([box]), M))[0]  
			box[box < 0] = 0
			img_crop = img_color[box[1][1]:box[0][1],box[1][0]:box[2][0]]
			cameras = camera_cascade.detectMultiScale(img_crop, scaleFactor=1.05, minNeighbors=5)
			fingerprint_sensor = fingerprint_sensor_cascade.detectMultiScale(img_crop, scaleFactor=1.05, minNeighbors=5)
			speaker = speaker_cascade.detectMultiScale(img_crop, scaleFactor=1.05, minNeighbors=5)
			if not isFeatureDetected(cameras) and not isFeatureDetected(fingerprint_sensor) and not isFeatureDetected(speaker):
				img_color = cv2.drawContours(img_color,[rect_box],0,(255,255,255),2)
	return img_color
	
def isFeatureDetected(cascade):
	for (_,_,_,_) in cascade:
		return True
	return False

# Use skeletonize from Scikit-image to get rid of spaces between close contours
def fillInContours(img):
	img = img_as_bool(img)
	img_out = ndi.distance_transform_edt(~img)
	img_out = img_out < 0.05 * img_out.max()
	img_out = morphology.skeletonize(img_out)
	img_out = morphology.binary_dilation(img_out, morphology.selem.disk(1))
	img_out = segmentation.clear_border(img_out)
	img_out = img_out | img
	return img_out

if __name__ == "__main__":
	main()
