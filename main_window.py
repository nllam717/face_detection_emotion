"""
In this example, we demonstrate how to create simple face detection using Opencv3 and PyQt5

Author: Berrouba.A
Last edited: 23 Feb 2018
"""

# import system module
import sys
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.uic import loadUi

# import Opencv module
import cv2

#from ui_main_window import *

class MainWindow(QDialog):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super(MainWindow,self).__init__()
        self.ui = loadUi('new.ui',self)
       
        # self.ui.setupUi(self)
        self.Return=0

        # load face cascade classifier
        self.emotion_classifier = load_model('model_filter2.h5', compile=False)
        self.EMOTIONS = ["Tuc gian","Kinh tom","So hai", "Hanh phuc", "Buon ba", "Bat ngo", "Binh thuong"]

        self.face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
        if self.face_cascade.empty():
            QMessageBox.information(self, "Error Loading cascade classifier" , "Unable to load the face	cascade classifier xml file")
            sys.exit()
        
        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.detectFaces)

        self.textBrowser.setText("Tức Giận")
        self.textBrowser_2.setText("Kinh Tởm")
        self.textBrowser_3.setText("Sợ hãi")
        self.textBrowser_4.setText("hạnh Phúc")
        self.textBrowser_5.setText("Buồn bã")
        self.textBrowser_6.setText("Bất ngờ")
        self.textBrowser_7.setText("Bình thường")
        # set control_bt callback clicked  function
        self.pushButton.clicked.connect(self.controlTimer)

    # detect face
    def detectFaces(self):
        # read frame from video capture
        ret, frame = self.cap.read()

        # resize frame image
        scaling_factor = 0.8
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # convert frame to GRAY format
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect rect faces
        face_rects = self.face_cascade.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30,30))
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        # for all detected faces
        if len(face_rects) > 0:
        # Chỉ thực hiện với khuôn mặt chính trong hình (khuôn mặt có diện tích lớn nhất)
            face = sorted(face_rects, reverse=True, key = lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = face
            # Tách phần khuôn mặt vừa tìm được và resize về kích thước 48x48 để chuẩn bị đưa vào bộ mạng Neural Network
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # Thực hiện dự đoán cảm xúc
            preds = self.emotion_classifier.predict(roi)[0]
            self.emotion_probability = np.max(preds)
            label = self.EMOTIONS[preds.argmax()]
            
            # Gán nhãn cảm xúc dự đoán được lên hình
            cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
            dem=0
            for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, preds)):
                
                text= "{}: {:.2f}%".format(emotion, prob * 100)   
                dem= dem + 1
                if dem==1:
                    self.textBrowser.setText(text)
                elif dem==2:
                    self.textBrowser_2.setText(text)
                elif dem==3:
                    self.textBrowser_3.setText(text)
                elif dem==4:
                    self.textBrowser_4.setText(text)
                elif dem==5:
                    self.textBrowser_5.setText(text)
                elif dem==6:
                    self.textBrowser_6.setText(text)
                elif dem==7:
                    self.textBrowser_7.setText(text)
                          
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                
        # self.label.setPixmap(cv2.imshow(canvas))
                

            
            
        # convert frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # canvas = cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)
        # get frame infos
        height, width, channel = frame.shape
        step = channel * width

        # height1, width1, channel1 = canvas.shape
        # step1 = channel1 * width1
        # create QImage from RGB frame
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        
        # show frame in img_label
        self.image_label.setPixmap(QPixmap.fromImage(qImg))
        # qImg = QImage(canvas.data,width1, height1, step1, QImage.Format_RGB888)
        # self.textBrowser.setPixmap(QPixmap.fromImage(qImg))
        
         



    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.pushButton.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.pushButton.setText("Start")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())