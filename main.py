from pdf2image import convert_from_path
import numpy as np
import pandas as pd
import pytesseract
from pytesseract import image_to_string
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract"
from PIL import Image
from cv2 import cv2
import os
from matplotlib import pyplot as plt
import string
from PIL import ImageFilter
import re
from flask import session

def hello():
    return "Hello"

poppler_path = r'C:\Program Files\poppler-0.68.0\bin'
# converting image based pdf to images
def pdf_image(file, fp=1, lp=1):
    return convert_from_path(file, 500, first_page=fp, last_page=lp, fmt='png', poppler_path=poppler_path)

#gaussian blurring
def gaussian_blur(image):
    gblur = cv2.GaussianBlur(image,(5,5),0)
    return gblur

# transforming image
def image_transformation(image):
    #converting image into gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # converting it to binary image by Thresholding
    # black and white, and inverted, because white pixels are treated as objects in contour detection
    threshold_img = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    # display image
    #show_scaled('threshold image', threshold_img)
    
    # using a kernel that is wide enough to connect characters but not text blocks, and tall enough to connect lines.
    #create a structuring element in the shape of rect - 8 x 8
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    #Closing is reverse of Opening, Dilation followed by Erosion.
    #It is useful in closing small holes inside the foreground objects, or small black points on the object.
    closing = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, kernel)
    #Dilation = expanding the pixels based on neighboring pixels
    dilate = cv2.dilate(threshold_img,kernel,iterations = 2)
    # display image
    #show_scaled('Dilated', dilate)
    return dilate


def find_contours(image):
    im2 = image.copy()
    dilate = image_transformation(im2)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    i = 1 # using a count
    mask = np.zeros(image.shape, dtype=np.uint8) # create an empty masking array with same size as the original image
    masks = [] # array for saving all the masked sections of the image
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt) # construct a bounding rectangle around the contours
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt) # contour area 
        if area > 4000 and aspect_ratio > .5:  # check and remove noise
            mask[y:y+h, x:x+w] = image[y:y+h, x:x+w] # mask the area detected by contours
            masks.append(image[y:y+h, x:x+w]) 
            cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) # build a rectangle over the image based on contours
            cv2.putText(im2,str(i),(x+w+10,y+h),0,1,(0,255,0)) # labelling the contours
            i= i+1
    # writing the contoured and masked image to file
    cv2.imwrite("trial/{0}.png".format("name"), im2)
    cv2.imwrite("trial/{0}_mask.png".format("name"), mask)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    data =  []
    masks.reverse()
    for mask in masks:
        data.append(pytesseract.image_to_string(mask, lang='eng', config='--psm 6'))
   # print(data)
    return data
    #return im2

def postprocess(data):
    return ""

def preprocess(f1):
    text = re.sub('[^A-Za-z0-9.,]+',' ',f1)
    text = re.sub(' +', ' ', text)
    text = text.lower()
    text = text.strip()
    return text

from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from textblob.en import Spelling 


def train_autocorrect():
    #training TextBlob with custom data
    textToLower = ""

    with open("trial/health_2.txt","r") as f1:           # Open our source file
        text = f1.read()                                  # Read the file                 
        textToLower = text.lower()                        # Lower all the capital letters
        words = re.findall("[a-z]+", textToLower)             # Find all the words and place them into a list    
        oneString = " ".join(words)                           # Join them into one string
        pathToFile = "trial/corpus.txt"                      # The path we want to store our stats file at
        spelling = Spelling(path = pathToFile)                # Connect the path to the Spelling object
        spelling.train(oneString, pathToFile) 

        # filehandler = open("pickle.txt", 'wb') 
        # session['spell_lib']=pickle.dump(spelling, filehandler)
        # initialize the redis connection pool
        # serialized = pickle.dumps(spelling)
        # filename = 'serialized.txt'

        # with open(filename,'wb') as file_object:
        #     file_object.write(serialized)

        # with open(filename,'rb') as file_object:
        #     raw_data = file_object.read()

        # deserialized = pickle.loads(raw_data)

    return True


def autocorrect(txt):
    final = []
    pathToFile = "trial/corpus.txt"  
    spelling = Spelling(path=pathToFile)
    txt = sent_tokenize(txt)
    if(not isinstance(txt, list)):
        txt = [txt]
    #txt = list(map(sent_tokenize, txt))
    for item in txt:
        text = ''
        txt = word_tokenize(item)
        for i in txt:
            text += ' ' + str(spelling.suggest(i)[0][0])
        final.append(text)
    return final


