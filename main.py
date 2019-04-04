import rawpy
import cv2
import argparse 
import numpy as np
from scipy import signal

def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def getRaw(path):
    raw = rawpy.imread(path)
    return raw.raw_image_visible

def colorFilterArray (data):
    shape = data.shape
    img = np.zeros([(shape[0]), (shape[1]), 3], dtype=np.double)
    
    for i in range(0, shape[0], 2):
        for j in range(0, shape[1], 2):
            img [i+1, j+1, 0]  = np.double( data[i+1, j+1])    #B
            img [i+1, j, 1]     =  np.double(data[i+1, j]) #G
            img [i, j+1, 1] =  np.double(data[i, j+1])     #G
            img [i, j, 2]   =  np.double(data[i, j])   #R /65535.0

    img = img*(1/65535.0)
    return img

def demosaicing(img, typeDemosaic):
    return bilinearDemosaicingOpt(img)

def bilinearDemosaicingOpt(img):
    maskG = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    maskRB = np.array([[0.25, 0.5, 0.25], [0.5, 0, 0.5], [0.25, 0.5, 0.25]])
    b = cv2.filter2D(img[:,:,0], -1, maskRB)
    g = cv2.filter2D(img[:,:,1], -1,maskG)
    r = cv2.filter2D(img[:,:,2], -1, maskRB)
    shape = img.shape
   
    #Recovery original values 
    for i in range(0, shape[0]-2, 2):
        for j in range(0, shape[1]-2, 2):
            b [i+1, j+1]  = img[i+1, j+1, 0]    #B
            g [i+1, j]    = img[i+1, j, 1] #G
            g [i, j+1] =  img[i, j+1, 1]     #G
            r [i, j]   =  img[i, j, 2]   #R /65535.0

    return cv2.merge((b,g,r))

def bilinearDemosaicing(img):
    imgFinal = img.copy()
    shape = img.shape

    for i in range(1, shape[0]-2):
        for j in range(1, shape[1]-2):
            if (img[i, j, 1] == 0):
                imgFinal[i, j, 1] = (img[i, j+1, 1] + img[i, j-1, 1] + img[i+1, j, 1] + img[i-1, j, 1])*0.25

            if (img[i, j, 2] == 0):
                if (img[i, j+1, 2] != 0 or img[i, j-1, 2] != 0):
                    imgFinal[i, j, 2] = (img[i, j+1, 2] + img[i, j-1, 2])*0.5
                elif(img[i+1, j, 2] != 0 or img[i-1, j, 2] != 0):
                    imgFinal[i, j, 2] = (img[i+1, j, 2] + img[i-1, j, 2])*0.5
                else:
                    imgFinal[i, j, 2] = (img[i-1, j+1, 2] + img[i+1, j-1, 2] + img[i-1, j-1, 2] + img[i+1, j+1, 2])*0.25
		    
            if (img[i, j, 0] == 0):
                if (img[i, j-1, 0] != 0 or img[i, j+1, 0] != 0):
                    imgFinal[i, j, 0] = (img[i, j+1, 0] + img[i, j-1, 0])*0.5
                elif(img[i+1, j, 0] != 0 or img[i-1, j, 2] != 0):
                    imgFinal[i, j, 0] = (img[i+1, j, 0] + img[i-1, j, 0])*0.5
                else:
                    imgFinal[i, j, 0] = (img[i-1, j+1, 0] + img[i+1, j-1, 0] + img[i-1, j-1, 0] + img[i+1, j+1, 0])*0.25
    return imgFinal[2:shape[0]-2, 2:shape[1]-2, :]

def balance_channel(channel, cutoff):
    low = np.percentile(channel, cutoff)
    high = np.percentile(channel, 100 - cutoff)
    new_channel = np.uint8(np.clip((channel-low)*255.0/(high - low), 0, 255))
    return new_channel

def automaticWhiteBalance(img, cutoff):
    b = balance_channel(img[:,:,0], cutoff)
    g = balance_channel(img[:,:,1], cutoff)
    r = balance_channel(img[:,:,2], cutoff)
    
    return cv2.merge((b,g,r))

def manualWhiteBalance (img):
    posX = 2307
    posY = 2289
    valR = img[posY, posX, 2]
    valG = img[posY, posX, 1]
    valB = img[posY, posX, 0]
     
    newChannelB = np.uint8(np.clip((img[:,:,0]*255.0/valB), 0, 255))
    newChannelG = np.uint8(np.clip((img[:,:,1]*255.0/valG), 0, 255))
    newChannelR = np.uint8(np.clip((img[:,:,2]*255.0/valR), 0, 255))
        
    return cv2.merge((newChannelB, newChannelG, newChannelR))

def whitePatch(img):
    maxR = np.max(img[:,:,2])
    maxB = np.max(img[:,:,0])
    maxG = np.max(img[:,:,1])

    while(maxG < 0.45):
        img[:,:,1] = img[:,:,1] * 2
        maxG = np.max(img[:,:,1])

    alpha = maxG/maxR
    beta = maxG/maxB

    newChannelB = np.uint8(np.clip((img[:,:,0]*255.0*beta), 0, 255))
    newChannelG = np.uint8(np.clip((img[:,:,1]*255.0), 0, 255))
    newChannelR = np.uint8(np.clip((img[:,:,2]*255.0*alpha), 0, 255))
 
    return cv2.merge((newChannelB, newChannelG, newChannelR))

def grayWorld(img):
    maxG = np.max(img[:,:,1])
    while(maxG < 0.45):
        img[:,:,1] = img[:,:,1] * 2
        maxG = np.max(img[:,:,1])
    
    avgR = np.mean(img[:,:,2])
    avgB = np.mean(img[:,:,0])
    avgG = np.mean(img[:,:,1])

    alpha = avgG/avgR
    beta = avgG/avgB

    newChannelB = np.uint8(np.clip((img[:,:,0]*255.0*beta), 0, 255))
    newChannelG = np.uint8(np.clip((img[:,:,1]*255.0), 0, 255)) 
    newChannelR = np.uint8(np.clip((img[:,:,2]*255.0*alpha), 0, 255))
    
    return cv2.merge((newChannelB, newChannelG, newChannelR))
    

def whiteBalance(img, typeWhiteBalance):
    if (typeWhiteBalance == "manual"):
        return manualWhiteBalance(img)
    elif (typeWhiteBalance == "whitePatch"):
        return whitePatch(img)
    elif (typeWhiteBalance == "grayWorld"):
        return grayWorld(img) 
    else:
        return automaticWhiteBalance(img, 1.0)

parser = argparse.ArgumentParser()
parser.add_argument("-path", type=str)
parser.add_argument("-gamma", default=2.2, type=float)
parser.add_argument("-whiteBalance", default="automatic", type=str)
parser.add_argument("-demosaic", default="bilinear", type=str)
args = parser.parse_args()

bayerData = getRaw(args.path) 
img = colorFilterArray(bayerData)

imgFinal = demosaicing(img, args.demosaic)
imgFinalWhiteBalance = whiteBalance(imgFinal, args.whiteBalance) #(balance_white(imgFinal, 2))

cv2.imwrite("foo.png",  imgFinalWhiteBalance)
cv2.imwrite("foo2.png",  adjust_gamma(imgFinalWhiteBalance, args.gamma))

