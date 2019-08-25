import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from PIL import Image
import glob

path = "./training/scene"

images = []
for f in glob.iglob(path):
    images.append(np.asarray(Image.open(f)))

def horizontal_flip(image_array: ndarray):
   #for performing data augementation -  horizontal image flipping w/o other rotations
    return image_array[:, ::-1]

images = np.array(images)
low_res[]

#low_res will only be used for detection purpose as the license plate detected may have irrelevant text due to augmentation
for im in images:
    imr = cv2.resize(im, (360, 360))
    blur = cv2.blulur(imr,(2,2),0) #image made blurry 
    blur = np.array(blur)
    low_res.append(blur)

#Augmentation
for im in images:
    imr = cv2.resize(im, (360, 360))
    blur = cv2.GaussianBlur(imr,(3,3),0) #image made blurry 
    blur = np.array(blur)
    rulb = horizontal_flip(im) #image flipped copy
    low_res.append(rulb)

df = data.copy(deep =True)
df_aug = data.copy(deep =True)


#Bounds Augmentation
for i in range(0,len(images)):
    length = abs(df_aug['points'][i][0]['x'] - df_aug['points'][i][1]['x'])
    df_aug['points'][i][0]['x'] = 0.5 + abs(0.5 - df_aug['points'][i][0]['x'])*np.sign(0.5 - df_aug['points'][i][0]['x']) - length
    df_aug['points'][i][1]['x'] = 0.5 + abs(0.5 - df_aug['points'][i][1]['x'])*np.sign(0.5 - df_aug['points'][i][1]['x']) + length

df = df.append(df_aug)
df.reset_index(drop = True ,inplace=True)

train_path = "./data/training/Preprocessing/"


for i in range(0,len(images)):

    img = Image.fromarray(low_res[i])

    img.save(train_path+'low_res/low_res'+str(i)+'.png')

    # Points of rectangle
    im = np.array(img)
    x1 = df['points'][i][0]['x']*im.shape[1]
    y1 = df['points'][i][0]['y']*im.shape[0]
    x2 = df['points'][i][1]['x']*im.shape[1]
    y2 = df['points'][i][1]['y']*im.shape[0]
    
for i in range(len(images),len(low_res)):

    img = Image.fromarray(low_res[i])

    img.save(train_path+'augmented/aug'+str(i)+'.png')

    # Points of rectangle
    im = np.array(img)
    x1 = df['points'][i][0]['x']*im.shape[1]
    y1 = df['points'][i][0]['y']*im.shape[0]
    x2 = df['points'][i][1]['x']*im.shape[1]
    y2 = df['points'][i][1]['y']*im.shape[0]

#saving the dataframe as new json file.
df.to_json(train_path+'data_net.json')