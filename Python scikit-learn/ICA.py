import mdp
import glob
import cv2
from sklearn import svm
import numpy as np

cv_img = []
for img in glob.glob('C:\Users\Choco\PycharmProjects\Task3\set/*.pgm'):
    n = cv2.imread(img,0).reshape((1,-1))
    cv_img.append(n)

arr = np.array(cv_img)
X = arr[:,0,:]

X = X.transpose()

y = mdp.fastica(X, dtype='float32')

y = y.transpose()

neg_c = np.zeros(250, dtype=np.int)
pos_c = np.ones(250, dtype=np.int)
b = np.hstack([neg_c,pos_c])

cv = svm.SVC(kernel='poly', probability=True, degree=5, coef0=0.1)
cv.fit(y,b)

counter = 0
counter0 = 0
counter1 = 0

for img in glob.glob("C:\Users\Choco\PycharmProjects\Task3\TrainImages/*.pgm"):
    n = cv2.imread(img,0).reshape((1,-1))
    if counter<250:
        if cv.predict(n)==0:
            counter0 = counter0+1
    if counter>250:
        if cv.predict(n)==1:
            counter1 = counter1+1
    counter = counter+1


accuracy = ((counter1+counter0)*100)/(counter)
print counter0, counter1, counter
print 'Accuracy of classifier is ' + str(accuracy) + '%.'

