import numpy as np
import sys
from sklearn import svm
import pickle
import glob
import cv2

cv_img = []
for img in glob.glob("database/train_py/*.jpg"):
    n = cv2.imread(img,0).reshape((1,-1))
    cv_img.append(n)

arr = np.array(cv_img)

X = arr[:,0,:]

c = np.zeros(365, dtype=np.int)
s = np.ones(360, dtype=np.int)
y = np.hstack([c,s])
#print y.shape
#y = [0,0,0,0,0,0,0,1,1,1]
cv = svm.SVC(kernel='poly', probability=True, degree=5, coef0=0.1)
cv.fit(X,y)

#t_img = cv2.imread("database/svm_test/CourierFont (222).jpg",0).reshape((1,-1))
#print(cv.predict(t_img))

counter = 0
counterc = 0
counters = 0

with open('classifier_py_n2.txt', 'wb') as fid:
    pickle.dump(cv, fid)


'''for img in glob.glob("database/test_py/*.jpg"):
    n = cv2.imread(img,0).reshape((1,-1))
    if counter<150:
        if cv.predict(n)==0:
            counterc = counterc+1
    if counter>=150:
        if cv.predict(n)==1:
            counters = counters+1
    counter = counter+1

accuracy = float((counterc + counters) * 100) / float(counter)

print (counter, counterc, counters, accuracy)'''