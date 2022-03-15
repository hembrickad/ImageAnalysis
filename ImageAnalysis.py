import os
import sys
import csv
import random 
import config as cfg
import numpy as np
import collections
import pandas as pd
from PIL import Image
from random import randint     
import matplotlib.pyplot as plt
from time import perf_counter

#Global Variables
cells = [[],[],[],[],[],[],[]]
histoTotal = [[],[],[],[],[],[],[]]
arrayTotal = [[],[],[],[],[],[],[]]
histoAVG = []
MSQE = [[],[],[],[],[],[],[]]
timeSheet = dict()

#Miscellaneous Methods
def intp(image, histo):
    for x in image:
        for n in x:
            n[0] = histo[n[0]] + (n[0] - (n[0] + 1))*((histo[n[0]+1]-histo[n[0]])/((n[0]+1)-n[0]))
            n[1] = n[0]
            n[2] = n[0]
    return image

def msqe(qArray, rArray):
    num = 0
    for x in range(len(rArray)):
        num += np.square(rArray[x] - qArray[x])
    return num/len(rArray)

#Set-up
def input(directory):
    for fil in os.listdir(directory):
        if fil.endswith(".BMP"):
            if fil.startswith("para"):
                cells[0].append(Image.open(directory + fil))
            elif fil.startswith("cyl"):
                cells[1].append(Image.open(directory + fil))
            elif fil.startswith("super"):
                cells[2].append(Image.open(directory + fil))
            elif fil.startswith("inter"):
                cells[3].append(Image.open(directory + fil))
            elif fil.startswith("let"):
                cells[4].append(Image.open(directory + fil)) 
            elif fil.startswith("mod"):
                cells[5].append(Image.open(directory + fil))
            elif fil.startswith("svar"):
                cells[6].append(Image.open(directory + fil))

def output(directory):
    cat = 0
    ver = 0
    for i in arrayTotal:
        for j in i:
            Image.fromarray(j).save(directory, "newPhotos",cat, "_", ver, ".BMP")
            ver += 1
        cat += 1
        ver = 0           

def TimeSheet(directory):
    df = pd.DataFrame.from_dict(timeSheet,orient='index')
    #df.columns['name': 'para', 'cyl', 'super', 'inter', 'let', 'mod', 'svar', 'total', 'average']
    df.to_csv(directory + 'TimeSheet.csv')

def HistoPrint(directory):
    cat = 0
    ver = 0
    for i in histoTotal:
        for j in i:
            for n in range(256):
                x.append(n)
            fig = plt.figure()  
            ax = fig.add_axes([0,0,1,1])
            ax.bar(x,histo)
            ax.set_facecolor('xkcd:black')
            plt.bar(x, histo, color=['white'])
            plt.tight_layout()
            plt.save(directory + 'histograms', cat, '_', ver, '.png')
            ver += 1
        cat += 1
        ver = 0

def config():
    histo = []
    im_arr = []

    if(cfg.config['DEFAULT']['pixel_val_grey'] == 'True'):
        timeSheet['grey'] = []
        total = 0

        for x in cells:
            t = perf_counter()
            for n in x:
                im_arr.append(pixel_val_grey(n, cfg.config['SETTINGS']['pixel_val_grey']))
            arrayTotal.append(im_arr.copy())
            im_arr = []

            total += (perf_counter() - t)
            timeSheet['grey'].append((perf_counter() - t))

        timeSheet['grey'].append(total)
        timeSheet['grey'].append(total/499)

    if(cfg.config['DEFAULT']['pixel_val_color'] == 'True'):
        timeSheet['color'] = []
        total = 0

        for x in cells:
            t = perf_counter()
            for n in x:
                im_arr.append(pixel_val_grey(n, cfg.config['SETTINGS']['pixel_val_color']))
            arrayTotal.append(im_arr.copy())
            im_arr = []

            total += (perf_counter() - t)
            timeSheet['color'].append((perf_counter() - t))

        timeSheet['color'].append(total)
        timeSheet['color'].append(total/499)

    if(cfg.config['DEFAULT']['binary'] == 'True'):
        timeSheet['binary'] = []
        total = 0

        for x in arrayTotal:
            t = perf_counter()
            for n in x:
                im_arr.append(binary(n))
            arrayTotal.append(im_arr.copy())
            im_arr = []

            total += (perf_counter() - t)
            timeSheet['binary'].append((perf_counter() - t))

        timeSheet['binary'].append(total)
        timeSheet['binary'].append(total/499)

    if(cfg.config['DEFAULT']['negative'] == 'True'):   
        timeSheet['negative'] = []
        total = 0

        for x in arrayTotal:
            t = perf_counter()
            for n in x:
                im_arr.append(negative(n))
            arrayTotal.append(im_arr.copy())
            im_arr = []

            total += (perf_counter() - t)
            timeSheet['negative'].append((perf_counter() - t))

        timeSheet['negative'].append(total)
        timeSheet['negative'].append(total/499)


    if(cfg.config['DEFAULT']['histo_one'] == 'True'):
        timeSheet['histo'] = []
        total = 0

        for x in arrayTotal:
            t = perf_counter()
            for n in x:
                histo.append(histo_one(n))
            histoTotal.append(histo.copy())
            histo = []

            total += (perf_counter() - t)
            timeSheet['histo'].append((perf_counter() - t))

        timeSheet['histo'].append(total)
        timeSheet['histo'].append(total/499)
    
    if(cfg.config['DEFAULT']['snp'] == 'True'):
        timeSheet['snp'] = []
        total = 0

        for x in arrayTotal:
            t = perf_counter()
            for n in x:
                n = snp(n,cfg.config['SETTINGS']['snp'])

            total += perf_counter() - t
            timeSheet['snp'].append(perf_counter() - t)
        timeSheet['snp'].append(total)
        timeSheet['snp'].append(total/499)

    if(cfg.config['DEFAULT']['gausNoise'] == 'True'):
        timeSheet['gausNoise'] = []
        total = 0

        for x in arrayTotal:
            t = perf_counter()
            for n in x:
                n = gausNoise(n,cfg.config['SETTINGS']['gausNoise'])

            total += perf_counter() - t
            timeSheet['gausNoise'].append(perf_counter() - t)
        timeSheet['gausNoise'].append(total)
        timeSheet['gausNoise'].append(total/499)

    if(cfg.config['DEFAULT']['speckle'] == 'True'):
        timeSheet['speckle'] = []
        total = 0

        for x in arrayTotal:
            t = perf_counter()
            for n in x:
                n = speck(n,cfg.config['SETTINGS']['speckle'])

            total += perf_counter() - t
            timeSheet['speckle'].append(perf_counter() - t)
        timeSheet['speckle'].append(total)
        timeSheet['speckle'].append(total/499)  

    if(cfg.config['DEFAULT']['histo_equal'] == 'True'):
        timeSheet['Equal'] = []
        total = 0

        for x in range(len(arrayTotal)):
            t = perf_counter()
            for y in range(len(arrayTotal[x])):
                arrayTotal[x][y] = histo_equal(arrayTotal[x][y], HistoTotal[x][y])

            total += (perf_counter() - t)
            timeSheet['Equal'].append((perf_counter() - t))

        timeSheet['Equal'].append(total)
        timeSheet['Equal'].append(total/499)

    if(cfg.config['DEFAULT']['histo_quant'] == 'True'):
        timeSheet['Quant'] = []
        total = 0
        oArrays = arrayTotal
        for x in range(len(arrayTotal)):
            t = perf_counter()
            for y in range(len(arrayTotal[x])):
                arrayTotal[x][y] = histo_quant(arrayTotal[x][y], histo_one(arrayTotal[x][y]))
            total += (perf_counter() - t)
            timeSheet['Quant'].append((perf_counter() - t))

        timeSheet['Quant'].append(total)
        timeSheet['Quant'].append(total/499)     

        print(timeSheet)

        for x in range(len(arrayTotal)):
            for n in range(len(arrayTotal[x])):
                MSQE[x][n] = msqe(arrayTotal,oArrays)

    if(cfg.config['DEFAULT']['MFilter'] == 'True'):
        timeSheet['MFilter'] = []
        total = 0
        oArrays = arrayTotal
        for x in range(len(arrayTotal)):
            t = perf_counter()
            for y in range(len(arrayTotal[x])):
                arrayTotal[x][y] = MFilter(arrayTotal[x][y], cfg.config['SETTINGS']['MFilter'])
            total += (perf_counter() - t)
            timeSheet['MFilter'].append((perf_counter() - t))

        timeSheet['MFilter'].append(total)
        timeSheet['MFilter'].append(total/499)

    if(cfg.config['DEFAULT']['histo_avg'] == 'True'):
        timeSheet['HistoAVG'] = []
        total = 0

        for x in histoTotal:
            t = perf_counter()
            histoAVG.append(histo_avg(x))

            total += perf_counter() - t
            timeSheet['HistoAVG'].append(perf_counter() - t)
        timeSheet['HistoAVG'].append(total)
        timeSheet['HistoAVG'].append(total/499)






#SPECTRUM MANIPULATION
def pixel_val_grey(image, channel = "k"):
    array = np.array(image)
    num = 0
    for x in array:
        for n in x:
            if channel == "r":
                n[1] = n[0]
                n[2] = n[0]
            elif channel == "g":
                n[1] = n[2]
                n[0] = n[2]
            elif channel == "b":
                n[0] = n[1]
                n[2] = n[1]
            else:
                num = (int(n[0])+int(n[1])+int(n[2]))/3
                n[0] = num
                n[1] = num
                n[2] = num
    return array

def pixel_val_color(image, channel = "r"):
    array = np.array(image)
    for x in array:
        for n in x:
            if channel == "r":
                n[1] = 0
                n[2] = 0
            elif channel == "b":
                n[1] = 0
                n[0] = 0
            elif channel == "g":
                n[0] = 0
                n[2] = 0
    return array

def binary(array, str = 127):
    for x in array:
        for n in x:
            if n[0] >=127:
                n[0] = 255
                n[1] = 255
                n[2] = 255
            else:
                n[0] = 0
                n[1] = 0
                n[2] = 0
    return array

def negative(array):
    for x in array:
        for n in x:
            n[0] = 255 - n[0]
            n[1] = 255 - n[1]
            n[2] = 255 - n[2]
    return array


#Noise
def snp(array, str = 10):
    num = 0
    if str <1: 
        return array
    else:
        for x in array:
            for n in x:
                r = randint(0, 100)
                if(r < str):
                    if(r %2 == 0):
                        n[0] = 0
                        n[1] = 0
                        n[2] = 0
                    else:
                        n[0] = 255
                        n[1] = 255
                        n[2] = 255
    return array 

def gausNoise(array, str = 10):
    num = 0
    if str == 0:
        return array
    else:
        for x in array:
            for n in x:
                num = randint(str*-1, str)
                n[0] += num 
                if n[0] < 0:
                    n[0] = 0
                elif n[0]> 255:
                    n[0] = 255
                n[1]=n[0]
                n[2]=n[0]
    return array

def speck(array, str = 75):
    num = 0
    if str == 0:
        return array
    elif str <= 100:
        for x in array:
            for n in x:
                num = random.uniform((str*-1)/100, str/100)
                n[0] *=(1+num)
                if n[0] < 0:
                    n[0] = 0
                elif n[0]> 255:
                    n[0] = 255
                n[1]=n[0]
                n[2]=n[0]
    return array

#Basic Histogram
def histo_one(array):
    histo = [0] * 256
    for x in array:
        for n in x:
            histo[n[0]] += 1
    return histo

def histo_avg(l):
    t = 0
    total= 0
    num = len(l)
    l = [sum(i) for i in zip(*l)]
    l = [n / num for n in l]

def histo_equal(array, histo):
    cd = [0] * len(histo)
    cd[0] = histo[0]
    for i in range(1, len(histo)):
        cd[i] = cd[i-1]+histo[i]
    cd = [x * 255 / cd[-1] for x in cd]
    arr = intp(array, cd) 
    return arr

def histo_quant(array, histo, str = 25):
    i = 0
    r = list(range(256))
    div = [r[x:x+str] for x in range(0, len(r), str)]

    div = [sum(i) for i in div]
    div = [round(n / str) for n in div]

    for x in div:
        for n in range(str):
            if(i > 255):
                break
            histo[i] = x
            i += 1

    for x in array:
        for n in x:
            n[0] = histo[n[0]]
            n[1] = n[0]
            n[2] = n[0]
    
    return array

#Filters           
def MFilter(array,median_filter = [[1, 2, 1],[2, 3, 2],[1, 2, 1]]):
    image_width, image_height = array.shape[1], array.shape[0]
    nArray = array.copy()

    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            # Get Image Filter Pixels
            filter_mid = len(median_filter)//2

            # Skip Border Pixels
            if x-filter_mid < 0 or x+filter_mid > image_width: continue
            if y-filter_mid < 0 or y+filter_mid > image_height: continue
            filter_pixels = array[y-filter_mid:y+filter_mid+1,x-filter_mid:x+filter_mid+1]

            # Apply Median Spatial Filter Over Greyscale Image
            # Store First Element (Pixel Value) In Fitler Pixel Row, Since
            #   They're All The Same Value (Greyscale Image)
            pixels = []

            for filter_x, rgb_pixel_values in enumerate(filter_pixels):
                for filter_y, rgb_values in enumerate(rgb_pixel_values):
                    pixel_weight = median_filter[filter_y][filter_x]
                    pixel_value  = rgb_values[0]
                    pixels.extend([pixel_value for _ in range(pixel_weight)])

            # Fetch Median Pixel Value Within Filter And Set New Image Pixel At (y,x) To New Median Value
            if len(pixels) > 1:
                pixels.sort()
                median = pixels[len(pixels)//2] if len(pixels) % 2 == 0 else pixels[len(pixels)//2+1]
                nArray[y][x][0] = median
                nArray[y][x][1] = nArray[y][x][0]
                nArray[y][x][2] = nArray[y][x][0]

    return nArray

def LFilter(array,  mean_filter = [[4, 5, 4],[5, 6, 5],[4, 5, 4]]):
    image_width, image_height = array.shape[1], array.shape[0]
    nArray = array.copy()

    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            # Get Image Filter Pixels
            filter_mid = len(mean_filter)//2

            # Skip Border Pixels
            if x-filter_mid < 0 or x+filter_mid > image_width: continue
            if y-filter_mid < 0 or y+filter_mid > image_height: continue
            filter_pixels = array[y-filter_mid:y+filter_mid+1,x-filter_mid:x+filter_mid+1]
            pixels = []

            for filter_x, rgb_pixel_values in enumerate(filter_pixels):
                for filter_y, rgb_values in enumerate(rgb_pixel_values):
                    pixel_weight = mean_filter[filter_y][filter_x]
                    pixel_value  = rgb_values[0]
                    pixels.extend([pixel_value for _ in range(pixel_weight)])

            # Fetch Mean of Pixel Value Within Filter And Set New Image Pixel At (y,x) To New Median Value
            if len(pixels) > 1:
                mean = int(round(np.mean(pixels)))
                nArray[y][x][0] = mean
                nArray[y][x][1] = nArray[y][x][0]
                nArray[y][x][2] = nArray[y][x][0]

    return nArray






def main():
    path = "/Users/Adhsketch/Desktop/repos/ImageAnalysis/cell_smears/inter01.BMP"
    #im_org = Image.open(path)
    HistList = []
    ArrayList = []
    #arr = pixel_val_grey(im_org)
    #arr = negative(arr)
    #hist = histo_one(arr)


    #Image.fromarray(arr).save( "NI.BMP")   

    #image = histo_quant(arr, hist, 75)

    #image = snp(arr, str = 25)

    #image = LFilter(arr)

    
    #Image.fromarray(image).save("NI.BMP")
    input(cfg.config['DEFAULT']['directory'])
    config()

    TimeSheet(cfg.config['DEFAULT']['directory'])

    #output(cfg.config['DEFAULT']['directory'])


    #HistoPrint(cfg.config['DEFAULT']['directory'])
    df = pd.DataFrame(MSQE)
    df.to_csv(directory + 'MSQE.csv')

    print(MSQE)
    print(timeSheet)

if __name__ == "__main__":
    main()   