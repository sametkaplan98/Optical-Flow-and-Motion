# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:09:34 2020

@author: Samet
"""
import cv2
import numpy as np
import imutils
import time
import math
import matplotlib.pyplot as plt

renk=(0,0,255)
video=cv2.VideoCapture(0)
MIN_ESLESME=25
aci_top=90
aci=0
fark=0
x_path2=200
y_path2=200
x_ort_eski=0
y_ort_eski=0
#x_ar=[]
#y_ar=[]
konsol_yol=np.zeros((1200,1200,3), dtype='uint8') # gecmis ekrani 
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
 
 t1=time.time()
 x_ort=0
 x_top=0
 y_ort=0
 y_top=0   
 sayac=1  

 
 if fark%3==0:              # gerçek goruntude 3 iyi
  _,frame1=video.read()
  fark=0
 a,frame2=video.read()
 
 color = (0,0,255)
 old_frame=frame1
 old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
 old_gray=cv2.GaussianBlur(old_gray,(9,9),0)
 p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
 mask = np.zeros_like(old_frame)
 frame = frame2
 frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 frame_gray=cv2.GaussianBlur(frame_gray,(9,9),0)
 p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)   
 good_new = p1[st==1]
 good_old = p0[st==1]
    
 for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()        
        x_top=x_top+(c-a)
        y_top=y_top+(d-b)
        mask = cv2.line(mask, (a,b),(c,d), renk, 2)
        frame = cv2.circle(frame,(a,b),2,renk,-1)
        sayac+=1
        
 x_ort=int(x_top/sayac)
 y_ort=int(y_top/sayac)     
 #x_ar.append(x_ort)
 #y_ar.append(y_ort)
 #plt.plot(range(len(y_ar)),y_ar,'ro') #kırmızı
 #plt.plot(range(len(x_ar)),x_ar,'bs')
 
 img = cv2.add(frame,mask)
 img=imutils.resize(img,800,800)
 
 sift=cv2.xfeatures2d.SIFT_create()
 kp1,des1=sift.detectAndCompute(old_gray,None)
 kp2,des2=sift.detectAndCompute(frame_gray,None) 
  
 if des2 is not None : #Yeterli çıkarımda bulunamadığı zaman hata verip
                       #programı sonlandırmaması için
  FLANN_INDEX_KDTREE=0
  index_params=dict(algorithm=FLANN_INDEX_KDTREE)  
  search_params=dict(checks=100)
  flann=cv2.FlannBasedMatcher(index_params,search_params)  
  matches=flann.knnMatch(des1,des2,k=2)
  good=[]   #Başarılı eşleştirmeler buraya atanacak
  
  for m,n in matches:
    if m.distance <0.7*n.distance:
     good.append(m)
  
  if len(good)>MIN_ESLESME :
   src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)    
   dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)                
   homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)    
   aci = math.atan2(homography[1,0], homography[0,0])*(180/math.pi)


  if abs(aci)>1:              #gercek goruntude 1 iyi
    aci_top=aci_top+aci
  if abs(aci_top)>=360:
     aci_top=0
  
  mesafe=math.sqrt((x_ort-x_ort_eski)*(x_ort-x_ort_eski)+(y_ort-y_ort_eski)*(y_ort-y_ort_eski))
  print(mesafe)

  if mesafe>=10:
   hareket_acisi=aci_top*math.pi/180
   x_path2=x_path2+(mesafe*math.cos(math.radians(aci_top)))
   y_path2=y_path2-(mesafe*math.sin(math.radians(aci_top)))
     
  cv2.circle(konsol_yol,(int((x_path2)*0.5+600),int((y_path2)*0.5+600)),1,renk,2)    
     
  cv2.putText(img,('ACI'),(50,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(0,0,255),2,cv2.LINE_AA)
  cv2.putText(img,(str(int(aci_top))),(110,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(0,0,255),2,cv2.LINE_AA)  
  img=imutils.resize(img,640,360)
  cv2.imshow('CANLI AKIS',img)   
  cv2.imshow('YOL',konsol_yol)
  old_gray = frame_gray.copy()
  p0 = good_new.reshape(-1,1,2)
   
  x_ort_eski=x_ort
  y_ort_eski=y_ort

  t2=time.time()
  #print('Sure:',(t2-t1))
  fark+=1
  if cv2.waitKey(1) &  0xFF == ord('q'):
   break

video.release()
cv2.destroyAllWindows()
