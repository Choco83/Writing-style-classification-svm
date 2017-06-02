import numpy as np
from svmutil import *
import glob
import cv2

neg_img = []
for img in glob.glob("database/test5/Courier/*.jpg"):
    n = cv2.imread(img,0).reshape((1,-1))
    neg_img.append(n)

pos_img = []
for img in glob.glob("database/test5/Segoe/*.jpg"):
    n = cv2.imread(img,0).reshape((1,-1))
    pos_img.append(n)

neg_arr = np.array(neg_img)
neg_arr = neg_arr[:,0,:]

#neg_arr = np.asarray(neg_arr).reshape(-1)
#neg_arr = np.array(neg_arr)

pos_arr = np.array(pos_img)
pos_arr = pos_arr[:,0,:]

#pos_arr = np.asarray(pos_arr).reshape(-1)
#pos_arr = np.array(pos_arr)

neg_c = np.zeros(250, dtype=np.int)
pos_c = np.ones(250, dtype=np.int)
y = np.hstack([neg_c,pos_c])

b = np.vstack((neg_arr,pos_arr))

Y1=[]
for i in range(0, b.shape[0]):
    Y1.append(list(b[i]))

prob = svm_problem(y,Y1)
param = svm_parameter('-t 1 -d 5 -r 0.1 -b 1')

m = svm_train(prob,param)
svm_save_model('moodle5.txt',m)