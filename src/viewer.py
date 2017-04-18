#!/usr/bin/env python
#-*-<encoding=UTF-8>-*-

import wx
import os
import cv2
import dlib
import pickle
import time
import openface
from sklearn.mixture import GMM
import numpy as np
import shutil
import time

status = "preview" # training 

class MainWindow(wx.Frame):
    
    def __init__(self,image,parent=None,id=-1,pos=wx.DefaultPosition, title="Face Recognidtion Viewer"):
        wx.Frame.__init__(self, parent, title=title, size=(960, 500))
        
        #=======================================================================
        self.capture = cv2.VideoCapture(0)
        #uncomment 2 lines below for opencv3
        #self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        #uncomment 2 lines below for opencv2
        self.capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
        #=======================================================================
        
        panel = wx.Panel(self, -1) # Make a new panel inside the frame
        panel.SetBackgroundColour('#ededed') # Set 'panels' bgcolor
 
        leftPan = wx.Panel(panel, -1) # Make a subpanel for the left of 'panel'
        leftPan.SetBackgroundColour('#4f5049') # Set 'leftPan' bgcolor
        
        rightPan = wx.Panel(panel , -1) # Make a subpanel for the right of 'panel'
        rightPan.SetBackgroundColour('#ededed') # Set 'rightPan' bgcolor
        
        vbox = wx.BoxSizer(wx.HORIZONTAL) # Make a new sizer for 'panel'

        vbox.Add(leftPan, 2, wx.EXPAND | wx.ALL, 10) # Add both subpanels to the new sizer \
        vbox.Add(rightPan, 1, wx.EXPAND | wx.ALL, 10) # in order

        panel.SetSizer(vbox) # set the sizer containing 'leftPan' and 'rightPan' 
        
        ShowCapture(leftPan, self.capture)

        gridSizer = wx.GridBagSizer(vgap=6, hgap=6)
        rightPan.SetSizer(gridSizer)
        self.labelInput = wx.TextCtrl(rightPan, -1)
        trainBtn = wx.Button(rightPan, -1, label="Train")
        self.updatePath = wx.TextCtrl(rightPan, -1, value = "192.168.100.1:/home/intel",)
        updateBtn = wx.Button(rightPan, label="Update")
        inferInfo = wx.StaticText(rightPan, -1)
        inferBtn = wx.Button(rightPan, -1, label="Infer")
              
        self.outputInfoPanel = wx.StaticText(rightPan)      
         
        gridSizer.Add(self.labelInput, pos=(0,0), span=(1,5), flag = wx.EXPAND | wx.ALL, border = 10)
        gridSizer.Add(trainBtn, pos=(0,5), span=(1,1), flag = wx.EXPAND | wx.ALL, border = 10)
        gridSizer.Add(self.updatePath, pos=(1,0), span=(1,5), flag = wx.EXPAND | wx.ALL, border = 10)
        gridSizer.Add(updateBtn, pos=(1,5), span=(1,1), flag = wx.EXPAND | wx.ALL, border = 10)
        gridSizer.Add(inferInfo, pos=(2,0), span=(1,5), flag = wx.EXPAND | wx.ALL, border = 10)
        gridSizer.Add(inferBtn, pos=(2,5), span=(1,1), flag = wx.EXPAND | wx.ALL, border = 10)
        gridSizer.Add(self.outputInfoPanel, pos=(3,0), span=(3,6), flag = wx.EXPAND | wx.ALL, border = 10)
        
        self.Bind(wx.EVT_BUTTON, self.train, trainBtn)
        self.Bind(wx.EVT_BUTTON, self.infer, inferBtn)
        self.Bind(wx.EVT_BUTTON, self.update, updateBtn)

        self.openface = OpenFace()
        
    def train(self, event):
        label = self.labelInput.GetValue()
        if label != "":
            if os.access("images/"+label, os.F_OK):
                shutil.rmtree("images/"+label)
            os.mkdir("images/"+label)

            num = 0
            while True:
                ret, frame = self.capture.read()
                if ret and len(self.openface.SaveImages(frame, label)) != 0:
                    num += 1
                if num >= 10:
                    break

            self.openface.Train()
            self.openface.RefreshPKL()

        else:
            msg = "Please input the label of training sets !"
            self.messageDlg(msg)
        
    def infer(self, event):    
        print "infer function"
        ret, frame = self.capture.read()
        if ret:
            person, confidence = self.openface.infer(frame)
            if person != "":
                self.outputInfoPanel.SetLabel("Predict {} with {:.2f} confidence.".format(person, confidence))
            else:
                self.outputInfoPanel.SetLabel("No people found...")
    
    def update(self, event):
        path = self.updatePath.GetValue()
        if path == '':
            msg = "Please input the target path !"
            self.messageDlg(msg)
            return
        self.outputInfoPanel.SetLabel("Updating classifier to the target device...")
        os.system('sshpass -p "ros" scp embeddings/classifier.pkl ros@'+path)

    def messageDlg(self, msg):
        dlg = wx.MessageDialog(self, msg, 'Warning', wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()
#===============================================================================
class ShowCapture(wx.Panel):
     def __init__(self, parent, capture, fps=15):
         wx.Panel.__init__(self, parent)
 
         self.capture = capture
         ret, frame = self.capture.read()
 
         height, width = frame.shape[:2]
     #    parent.SetSize((2*width, height))
         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
         self.bmp = wx.BitmapFromBuffer(width, height, frame)
 
         self.timer = wx.Timer(self)
         self.timer.Start(1000./fps)
 
         self.Bind(wx.EVT_PAINT, self.OnPaint)
         self.Bind(wx.EVT_TIMER, self.NextFrame)
         
     def OnPaint(self, evt):
         dc = wx.BufferedPaintDC(self.GetParent())
         dc.DrawBitmap(self.bmp, 0, 0)
 
     def NextFrame(self, event):
         ret, frame = self.capture.read()
         if ret:
             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
             self.bmp.CopyFromBuffer(frame)
             self.Refresh()


class OpenFace(object):
    """docstring for OpenFace"""
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.align = openface.AlignDlib("../models/dlib/shape_predictor_68_face_landmarks.dat")
        self.net = openface.TorchNeuralNet("../models/openface/nn4.small2.v1.t7", 96, False)
        with open("embeddings/classifier.pkl", 'r') as f:
            (self.le, self.clf) = pickle.load(f) 
    
    def RefreshPKL(self):
        with open("embeddings/classifier.pkl", 'r') as f:
            (self.le, self.clf) = pickle.load(f) 

    def Train(self):
        os.system("rm -rf aligned-images/*")
        os.system("./align-dlib.py images/ align outerEyesAndNose aligned-images/ --size 96")
        os.system("rm aligned-images/cache.t7")
        os.system("./main.lua -outDir embeddings/ -data aligned-images/")
        os.system("./classifier.py train embeddings/")

    def Update(self):
        pass

    def SaveImages(self, frame, name):
        detected_faces = self.detector (frame, 1)
        if len(detected_faces) !=0 :
            cv2.imwrite("images/{}/{}-{}.jpg".format(name, name, time.clock()), frame)
        return detected_faces
    
    def getRep(self, rgbImg):
        start = time.time()


        bb1 = self.align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
        if bb1 is None :
            print("Unable to find a face")
            return []

        reps = []
        for bb in bbs:
            start = time.time()
            alignedFace = self.align.align(
                96,
                rgbImg,
                bb,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                print("Unable to align image")

            start = time.time()
            rep = self.net.forward(alignedFace)
            reps.append((bb.center().x, rep))
        sreps = sorted(reps, key=lambda x: x[0])
        return sreps

    def infer(self, frame):
        reps = self.getRep(frame)
        if len(reps) == 0:
            return "",0
        if len(reps) > 1:
            print("List of faces in image from left to right")
        for r in reps:
            rep = r[1].reshape(1, -1)
            bbx = r[0]
            start = time.time()
            predictions = self.clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = self.le.inverse_transform(maxI)
            confidence = predictions[maxI]
            print("Predict {} with {:.2f} confidence.".format(person, confidence))
            if isinstance(self.clf, GMM):
                dist = np.linalg.norm(rep - clf.means_[maxI])
                print("+ Distance from the mean: {}".format(dist))             
        return person, confidence

if __name__ == "__main__":

    app = wx.App()
    mainWin = MainWindow(None, title = "Face Recognition Viewer")

    mainWin.Show()
    app.MainLoop()
