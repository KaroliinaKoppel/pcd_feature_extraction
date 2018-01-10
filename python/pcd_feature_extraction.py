import sys
import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from skimage import io, morphology, img_as_bool, segmentation, img_as_ubyte
from scipy import ndimage as ndi

def main():
    img = readImage(sys.argv[1])
    plt.imshow(img)
    plt.show()
    canny = findCannyEdges(img)
    contours = findContours(canny)
    plt.imshow(contours)
    plt.show()

def readImage(fullname):
    filename, extension = os.path.splitext(fullname)
    if (extension != '.png'):
        print('File type must be .png')
        return -1
    img = cv2.imread(fullname)
    return img

def findCannyEdges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    canny = cv2.Canny(gray,100,200)
    return np.uint8(canny)

def findContours(img):
    kernel = np.ones((5,5),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 1)
    img = img_as_ubyte(fillInContours(img))
    _,contours,hierarchy = cv2.findContours(img, 1, 2)
    basename = sys.argv[1]
    filename, extension = os.path.splitext(sys.argv[1])
    img = readImage(basename)
    img_color = readImage(basename)
    count = 0
    rows,cols = img.shape[0], img.shape[1]
    for cnt,hier in zip(contours,hierarchy[0]):
        area = cv2.contourArea(cnt)
        if (area > 5):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            rect_box = np.int0(box)
            img_color = cv2.drawContours(img_color,[rect_box],0,(0,0,255),2)
            angle = rect[2]
            M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            box = np.int0(cv2.transform(np.array([box]), M))[0]  
            box[box < 0] = 0
            img_crop = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[box[1][1]:box[0][1], 
                       box[1][0]:box[2][0]]
            cv2.imwrite(filename+'_'+str(count)+extension, img_crop)
            count += 1
    return img_color

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
