import os
import re
import sys
import csv
import math
import random 
import pathlib
import numpy as np
import collections
import pandas as pd
import config as cfg
from PIL import Image
from random import randint
import ImageAnalysis as p1
import ImageAnalysis_2 as p2
from time import perf_counter    
import matplotlib.pyplot as plt

### Global ###
dataset = list()

### KNN ###
'''
def EuDist(r1, r2):
    d = 0.0
    for x in range(len(r1)-1):
        d += (r1[x] - r2[x])**2
    return sqrt(d)

def getN(train, testR, neighbors):
    d = []
    n = []
    for t in train:
       Edist = EuDist(train, test)
       d.append().append(train, Edist)
    d.sort(key=lambda tup: tup[1])
    for i in range(neighbors):
        n.append(d[i][0])
	return n

def predict(train, testR, neighbors):
    n = getN(train, testR, neighbors)
    output = [row[-1] for row in neighbors]
    return max(set(output), key=output.count)
'''
#### 10-Fold Cross Validation #####




####Features#####
def perimeter(image):
    p = 0 
    image = p2.Border_Detect(image)
    for x in image:
        for y in x:
            if y[0] != 0:
                p += 1
    return p   

def area(image):
    a = 0
    for x in image:
        for y in x:
            if y[0] != 255:
                a += 1
    return a 

def roundness(image, p, a):
    r = (p*p)/(4* math.pi *a)
    return r

def mode(hist):
    high = 0
    mode = 0
    for x in range(len(hist)):
        if hist[x] > high and x != 255:
            high = hist[x]
            mode = x
    return mode

####Cosmetics####
def data(image,p,a,r,m):
    #header = ['perimeter', 'area', 'roundness', 'mode', 'class']
    name = pathlib.PurePosixPath(image).stem
    name = re.sub(r'[0-9]+','',name)
    d =list((p,a,r,m,name))
    dataset.append(d)

def csv():
    df = pd.DataFrame(dataset)
    df.to_csv('/Users/Adhsketch/Desktop/repos/ImageAnalysis/Feature_Extraction.csv', header = ['perimeter', 'area', 'roundness', 'mode', 'class'])

def input(directory):
    for fil in os.listdir(directory):
        if fil.endswith(".BMP"):
            im_org = Image.open(directory+fil)
            arr = p1.pixel_val_grey(im_org)
            arr = p2.balanced(arr)

            hist = p1.histo_one(arr)

            p = perimeter(arr)
            a = area(arr)
            r = roundness(arr,p ,a)
            m = mode(hist)

            data(directory+fil,p,a,r,m)
    csv()
            


#"/Users/Adhsketch/Desktop/repos/ImageAnalysis/cell_smears/let51.BMP"
def main():
    directory = "/Users/Adhsketch/Desktop/repos/ImageAnalysis/Test/"
    input(directory)
    '''im_org = Image.open(directory)
    arr = p1.pixel_val_grey(im_org)
    arr = p2.balanced(arr)
    Image.fromarray(arr).save("NI.BMP")

    hist = p1.histo_one(arr)

    p = perimeter(arr)
    a = area(arr)
    r = roundness(arr,p ,a)
    m = mode(hist)

    data(directory, p, a, r, m)
    csv()
'''


    #

if __name__ == "__main__":
    main()   



    