from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
import numpy as np
from sklearn import svm
from tkinter.filedialog import askopenfilename
from sklearn.externals import joblib
from matplotlib import pyplot as plt


def get_image():
    global realImage,s_and_p,medimg,path,imgPanelA,tImg,negImage,new,svd_img
    path = askopenfilename()
    if len(path) > 0:
        realImage = cv2.imread(path)
        medimg = cv2.imread(path,0).reshape((1,-1))
        medimg = cv2.resize(medimg, (320, 240)).reshape((1, -1))
        tImg = cv2.imread(path,1)
        svd_img = Image.open(path)
        b,g,r = cv2.split(realImage)
        realImage = cv2.merge((r,g,b))
        new = realImage
        negImage = -realImage
        realImage = Image.fromarray(realImage)
        negImage = Image.fromarray(negImage)
        realImage = ImageTk.PhotoImage(realImage)
        negImage = ImageTk.PhotoImage(negImage)
    if imgPanelA is None:
    # storing of browsed image
        imgPanelA = Label(image=realImage)
        imgPanelA.image = realImage
        imgPanelA.pack(side='left', padx=10, pady=10)
    else:
        imgPanelA.configure(image=realImage)
        imgPanelA.image = realImage

    j = model.predict(medimg)
    if j==0:
        print("Courier Detected!!")
    elif j==1:
        print("Segoe Detected!!")

def webcam():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "test.jpg"
            cv2.imwrite(img_name, frame)
            #print("{} written!".format(img_name))
            #img_counter += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(frame, (320, 240)).reshape((1,-1))
    j = model.predict(resized_image)

    if j == 1:
        print("Segoe Detected!!")
    elif j == 0:
        print("Courier Detected!!")

    cam.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

model = joblib.load('classifier_py_n.txt')
main = Tk()
main.configure(background='midnightblue')
main.configure()
imgPanelA = None

browse_btn = Button(main, text="Browse Image", command=get_image)
browse_btn.pack(side = 'top',expand = "no",pady=0)

cam_btn = Button(main, text="WebCam", command=webcam)
cam_btn.pack(side = 'top',expand = "no",pady=0)

main.mainloop()
