import numpy as np
import pandas as pd 
import requests
import json
import random
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import urllib

import os

# Read the json file to extract images from source urls
data = pd.read_json('./Indian_Number_plates.json', lines=True)
pd.set_option('display.max_colwidth', -1)

# Delete the empty column
del data['extras']

# Extract the points of the bounding boxes because thats what we want
data['points'] = data.apply(lambda row: row['annotation'][0]['points'], axis=1)

# And drop the rest of the annotation info
del data['annotation']

train_path = "./training/"
def downloadData(df):
    i=0
    for index, row in df.iterrows():

        # Get the image from the URL and save the image
        resp = urllib.request.urlopen(row[0])
        img = Image.open(resp)

        img.save(train_path+'scene/car'+str(i)+'.png')

        # Points of rectangle
        im = np.array(img)
        x1 = row[1][0]['x']*im.shape[1]
        y1 = row[1][0]['y']*im.shape[0]
        x2 = row[1][1]['x']*im.shape[1]
        y2 = row[1][1]['y']*im.shape[0]

        # Cut the plate from the image and use it as output
        car = Image.fromarray(im)
        roi = car.crop((x1, y1, x2, y2))
        roi.save(train_path+'plate/plate'+str(i)+'.png')
        #Plates.append(np.array(roi))
        i+=1

downloadData(data)