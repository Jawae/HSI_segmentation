import numpy as np
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

predicted_matrix = model.predict(X_test) #inference: use trained u-net to make predictions about test image


def compute_segmentation_final(predicted_matrix):
    # initials values
    classes = list(range(1,7))
    colour = [[0,0,255],[0,100,0],[128,128,0],[255,255,0],[128,128,128],[210,180,140]] #rgb colour codes
    classnames = ['Pine Tree','Broadleaf Tree','Grass','Sand','Asphalt','Artificial Grass'] #classnames, e.g. for legend

    pred = predicted_matrix[0]
    [nclass, x, y] = pred.shape

    img = np.ones((x,y,3), dtype=np.uint8)*255
    for i in range(0,nclass):
        for j in range(0,x):
            for k in range(0,y):
                vector = pred[:,j,k]
                max = np.amax(vector) #take class with highest prediction-score
                if vector[i]==max:
                    img[j,k,:] = colour[i]

    # Predicted image
    save_img = Image.fromarray(img)
    save_img.save('Output/Segmentation.png')

    return
