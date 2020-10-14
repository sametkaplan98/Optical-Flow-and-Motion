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

renk=(0,0,255)
video=cv2.VideoCapture(0)
MIN_ESLESME=25
aci_top=90
aci=0
fark=0
x_path2=600 #default olarak x ve y'nin orjini
y_path2=600
x_ort_eski=0
y_ort_eski=0

konsol_yol=np.zeros((1200,1200,3), dtype='uint8') # gecmis yol ekrani 

feature_params = dict( maxCorners = 100, 
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
 x1=0
 x2=0
 y1=0
 y2=0
 t1=time.time() #calisma hizi olcumu
 x_ort=0
 y_ort=0 
 sayac=1  
 
 if fark%3==0: 
  _,frame1=video.read()
  fark=0
 _,frame2=video.read() 
 
 eski_frame=frame1                      
 eski_gri = cv2.cvtColor(eski_frame, cv2.COLOR_BGR2GRAY)
 eski_gri=cv2.GaussianBlur(eski_gri,(9,9),0)
 p0 = cv2.goodFeaturesToTrack(eski_gri, mask = None, **feature_params)
 mask = np.zeros_like(eski_frame)  #goruntulerin maske olusturularak cizdirilmesi icin
 frame = frame2
 frame_gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 frame_gri=cv2.GaussianBlur(frame_gri,(9,9),0)
 p1, st, err = cv2.calcOpticalFlowPyrLK(eski_gri, frame_gri, p0, None, **lk_params)   
 basarili_yeni = p1[st==1]
 basarili_eski = p0[st==1]
    
 for i,(yeni,eski) in enumerate(zip(basarili_yeni, basarili_eski)):
   
        a,b = yeni.ravel()
        c,d = eski.ravel()        
        x1=x1+a
        x2=x2+c
        y1=y1+b
        y2=y2+d
        mask = cv2.line(mask, (a,b),(c,d), renk, 2)
        frame = cv2.circle(frame,(a,b),2,renk,-1)
        sayac+=1 # kac adet nokta bulundugu   
     
 x_ort=int((x2)/sayac)
 y_ort=int((y2)/sayac)    
 x_ort_eski=int((x1)/sayac)
 y_ort_eski=int((y1)/sayac) 

 img = cv2.add(frame,mask)
 img=imutils.resize(img,800,800)
 
 sift=cv2.xfeatures2d.SIFT_create()
 kp1,des1=sift.detectAndCompute(eski_gri,None)
 kp2,des2=sift.detectAndCompute(frame_gri,None) 
  
 if des2 is not None : #Yeterli çıkarımda bulunamadığı zaman hata verip
                       #programı sonlandırmaması için
  FLANN_INDEX_KDTREE=0
  index_params=dict(algorithm=FLANN_INDEX_KDTREE)  
  search_params=dict(checks=100)
  flann=cv2.FlannBasedMatcher(index_params,search_params)  
  matches=flann.knnMatch(des1,des2,k=2)
  good=[]   #Başarılı eşleştirmeler buraya atanacak
  
  for m,n in matches:
    if m.distance <0.7*n.distance: #descriptorlar arasi mesafe               
     good.append(m)
  
  if len(good)>MIN_ESLESME :
   src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)    
   dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)                
   homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)    
   aci = math.atan2(homography[1,0], homography[0,0])*(180/math.pi)

  if abs(aci)>1:        #gercek goruntude 1, gurultu acilari engeller
    aci_top=aci_top+aci
  if abs(aci_top)>=360:
     aci_top=0
  
  mesafe=math.sqrt((x_ort-x_ort_eski)*(x_ort-x_ort_eski)+(y_ort-y_ort_eski)*(y_ort-y_ort_eski))

  if mesafe>=3:  # aci degisimlerinin ve gurultunun translasyon olarak alginmasinin azaltilmasi
   hareket_acisi=aci_top*math.pi/180
   x_path2=x_path2+(mesafe*math.cos(math.radians(aci_top)))
   y_path2=y_path2-(mesafe*math.sin(math.radians(aci_top)))
     
  cv2.circle(konsol_yol,(int((x_path2)*0.5),int((y_path2)*0.5)),1,renk,2)         
  cv2.putText(img,('ACI'),(50,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(0,0,255),2,cv2.LINE_AA)
  cv2.putText(img,(str(int(aci_top))),(110,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(0,0,255),2,cv2.LINE_AA)  
  img=imutils.resize(img,640,360)
  cv2.imshow('CANLI AKIS',img)   
  cv2.imshow('YOL',konsol_yol)
  
  eski_gray = frame_gri.copy()
  p0 = basarili_yeni.reshape(-1,1,2)


  t2=time.time()
  #print('Sure:',(t2-t1))
  fark+=1
  if cv2.waitKey(1) &  0xFF == ord('q'):
   break

video.release()
cv2.destroyAllWindows()