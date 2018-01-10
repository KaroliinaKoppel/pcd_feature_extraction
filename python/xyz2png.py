import sys
import os
from PIL import Image, ImageDraw
import cv2
import numpy as np

def main():
    file, filename = readFile()
    file_lines = file.readlines()
    Lines = list(file_lines)
    processFile(Lines, filename)

def readFile():
    if (len(sys.argv) != 2):
        print('Usage: python xyz2png.py <filename>')
        return -1
    filename, extension = os.path.splitext(sys.argv[1])
    if (extension != '.txt'):
        print('File type must be .txt')
        return -1
    return (open(filename + extension, 'r'), filename)

def processFile(Lines, filename):
    x_max = -sys.maxsize
    x_min = sys.maxsize
    y_max = 0
    z_max = -sys.maxsize
    z_min = sys.maxsize
    for line in Lines:
        x, y, z = list(map(int, line.split(' ')))
        if (x > x_max):
            x_max = x
        if (x < x_min):
            x_min = x
        if (y > y_max):
            y_max = y
        if (z > z_max):
            z_max = z
        if (z < z_min):
            z_min = z
    return buildImage(x_max - x_min, y_max, x_min, z_min, z_max, Lines, filename)

# Draw image using Pillow 
def buildImage(x, y, x_min, z_min, z_max, Lines, filename):
    basename = os.path.join(filename, ".png")
    im = Image.new("RGB", (x, y))
    draw = ImageDraw.Draw(im)
    for line in Lines:
        x, y, z = list(map(int, line.split(' ')))
        rgb = z2rgb(z_min, z_max, z)
        draw.point((x - x_min, y), rgb)
    del draw
    return cropImage(im, basename)

# Convert Z to RGB using minimum and maximum value of Z
def z2rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b

def cropImage(im, filename):
    # Convert Pillow image to OpenCV image
    img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur image to blur out details
    blur = cv2.GaussianBlur(gray,(15,15),0)

    #Use Morphological Closing and thresholding to get shape of object
    kernel = np.ones((15,15),np.uint8)
    closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    _, th = cv2.threshold(closing,0,255,0)

    # Get minimal object bounding box
    im2,contours,hierarchy = cv2.findContours(closing,1,2)
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Rotate image to minimize background
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    rot = cv2.warpAffine(img,M,(cols,rows))
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # Crop and save image
    crop = rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]
    cv2.imwrite(filename, crop)

if __name__ == "__main__":
    main()

    
