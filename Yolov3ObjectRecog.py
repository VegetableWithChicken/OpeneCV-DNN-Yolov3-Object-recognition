#! -*-coding: utf-8 -*-
import cv2
import numpy as np
import os
import configparser
import datetime
import socket
import sys

#读取当前文件设置信息
cf=configparser.ConfigParser()
#如果当前环境在ide 修改下面一行代码如下
#configDir=os.path.dirname(os.path.realpath(__file__))+"/TrainingModel/"
configDir=os.path.dirname(os.path.realpath(sys.executable))+"/TrainingModel/"
print(configDir)
cf.read(configDir+"Config.ini")
print(cf)
classesFile=configDir+"coco.names"
Yolov3Config=configDir+"yolov3.cfg"
Yolov3Weights=configDir+"yolov3.weights"

#读取可以检测的对象类型名称
classNames=[]
with open(classesFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')

secf=cf.sections()
dayConfidence=(float)(cf.get("ConfigData","dayConfidence"))#白天的置信度
nightConfidence=(float)(cf.get("ConfigData","nightConfidence"))
alphaValue=(float)(cf.get("ConfigData","alphaValue"))
BrightlessValue=(float)(cf.get("ConfigData","BrightlessValue"))
Yolo3Confidence=(float)(cf.get("ConfigData","Yolo3Confidence"))
nms_Yolo3Confidence=(float)(cf.get("ConfigData","nms_Yolo3Confidence"))

#设置网络层信息
ip=cf.get("ConfigData","ip")
port=(int)(cf.get("ConfigData","port"))
listener=int(cf.get("ConfigData","listener"))
listener=listener if listener>0 else 10
#设置TCP网络服务器
tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_socket.bind((ip,port))
tcp_socket.listen(listener)
client, addr = tcp_socket.accept()

net=cv2.dnn.readNetFromDarknet(Yolov3Config,Yolov3Weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layerNames=net.getLayerNames()
outputNames=[layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

#检测对象信息
def findObject(outputs,img):
    height, width, channel = img.shape
    boundingBox = []
    classIDs = []
    confidences = []
    for outputs in outputs:
        for det in outputs:
            scores = det[5:] # Confidence value of 80 classes
            classID = np.argmax(scores) 
            confidence = scores[classID]
            if confidence>Yolo3Confidence:
                w, h = int(det[2]*width), int(det[3]*height)  #Box Width and Height
                x, y = int((det[0]*width) - w/2), int(det[1]*height - h/2)  # Center point
                boundingBox.append([x, y, w, h])
                classIDs.append(classID)
                confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(boundingBox, confidences, Yolo3Confidence, nms_Yolo3Confidence)
    for num in indices:
       index=num[0]
       prename=classNames[classIDs[index]]
       if prename=="person":
           #向UE4 发送程序
            st=prename.encode('gbk')
            if client:
              client.send(st)
        

#运行程序
def StartRun(frame):
    blob=cv2.dnn.blobFromImage(frame,1/255,(608,608),[0,0,0],crop=False)
    net.setInput(blob)
    outputs=net.forward(outputNames)
    findObject(outputs,frame)

#增加图片亮度和对比度
def increased(alpha,brightless,frame):
    rows,cols,channel=frame.shape
    for i in range(rows):
        for j in range(cols):
            for c in range(3):
                color=frame[i,j][c]*alpha+brightless
                if color>255:
                    frame[i,j][c]=255
                elif color<0:
                    frame[i,j][c]=0



if __name__ == "__main__":
    Cap=cv2.VideoCapture(0)
    while Cap.isOpened():
        success,img=Cap.read()
        h=datetime.datetime.now().hour
        if h>19|h<7:
            increased(alphaValue,BrightlessValue,img)
        StartRun(img)
client.close()




