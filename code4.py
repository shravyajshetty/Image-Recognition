from PIL import Image
import numpy, os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm , metrics
import cv2
from skimage import data, io, filters
from skimage.filters import threshold_local
from skimage.filters import threshold_otsu, threshold_adaptive
path="/home/grim/learning_image_classification/code/"
Xlist=[]
Ylist=[]
for directory in os.listdir(path):
    for file in os.listdir(path+directory):
        print(path+directory+"/"+file)
        img=Image.open(path+directory+"/"+file)
        featurevector=numpy.array(img).flatten()
        Xlist.append(featurevector)
        Ylist.append(directory)
classifier = svm.SVC(gamma = 0.001 , C = 100)
scores = cross_val_score(classifier, Xlist, Ylist,cv=3)
file = open("output.txt" , "w+")
file.write(str(scores))
score1 = scores.mean()


Xlist2=[]
Ylist2=[]
for directory in os.listdir(path):
    for file in os.listdir(path+directory):
        print(path+directory+"/"+file)
        img=cv2.imread(path+directory+"/"+file , 0)
        global_thresh = threshold_otsu(img)
        block_size = 35
        adp_img = threshold_local(img, block_size, offset=15)
        img = img > adp_img
        featurevector=numpy.array(img).flatten()
        Xlist2.append(featurevector)
        Ylist2.append(directory)
classifier2 = svm.SVC(gamma = 0.001 , C = 100)
scores = cross_val_score(classifier2, Xlist2, Ylist2,cv=3)
file = open("output_adaptive.txt" , "w+")
file.write(str(scores))
file.close()
print(score1)
print(scores.mean())
