# -*- coding: utf-8 -*-
"""
Created on Wed May 29 23:14:12 2019

@author: reuve
"""

import os
import imageio
import numpy as np
#from skimage import io, color
from skimage import color
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from time import time
# from sklearn.model_selection import GridSearchCV

# path to the data directory
path = r'faces94/female'

# This would print all the files and directories
# all the images are 180X200 pixels.
cols = 180
rows = 200
area = cols*rows
 
images = {}
i = 0    
 
# loading the images in a dictionary
for subdir, dirs, files in os.walk(path):
    # for each file
    for file in files:
        image_file_path = subdir + os.sep + file
#        print(image_file_path)
        if image_file_path.endswith(".jpg"):
            face = imageio.imread(image_file_path)
            # each pixel in 'face' has RGB values. need to transfer it to gray levels
            images[file] = color.rgb2gray(face).reshape([1, cols*rows])
            i += 1
print('read {} files'.format(i))    

data = np.zeros([len(images), cols*rows])
y = []
i = 0

for image in images:
    y.append(image[:image.find('.')])
    data[i,:] = images[image]
    i += 1

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)
n_components = 30
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
pca.fit(X_train)

for i in range(4):
    eigenface = pca.components_[i].reshape([rows , cols])
    plt.imshow(eigenface)
    plt.show()
    
eigenfaces = pca.components_.reshape([n_components, rows, cols])
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

print("Fitting the classifier to the training set")

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        plt.show()

#h= rows
#w=180
# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, i):
    pred_name = y_pred[i]
    true_name = y_test[i]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


#from numpy import linalg as LA

i=0
AA = np.empty(shape=( len(images),area), dtype='float64')
for image in images:
    AA[i,:] = images[image]
    i+=1
    
l=len(AA)
mean_image = np.sum(AA, axis=0)/l
for k in range(l):
    AA[k,:] -= mean_image
    
 
Cov = np.matrix(AA)*np.matrix(AA.transpose())
Cov /= l
 
eigenVal, eigenVec = np.linalg.eig(Cov)
sort_indices = eigenVal.argsort()[::-1]                             
eigenVal = eigenVal[sort_indices]                               
eigenVec = eigenVec[sort_indices]
 
eigenVal_sum = sum(eigenVal[:])
eigenVal_energy = 0.0
energy = 0.80

i = 0
for eigenv in eigenVal:
    i+=1
    eigenVal_energy += eigenv / eigenVal_sum
    if eigenVal_energy >= energy:
        break
 
eigenVal = eigenVal[0:i]
eigenVect = eigenVec[0:i]
  
eigenFaces = np.dot( AA.transpose(),eigenVect.transpose())
norms = np.linalg.norm(eigenVect, axis=1)
eigenFaces = eigenFaces / norms
 
W = AA*eigenFaces
eigenFaces = eigenFaces

