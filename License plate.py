import torch
import cv2
import easyocr
import time
from PIL import Image, ImageTk
import datetime
import imutils
from imutils.video import VideoStream
import numpy as np
from playsound import playsound 
import vlc 
from smtplib import SMTP
import threading




#model = torch.hub.load('C:\\Users\\yaman\\Desktop\\yolov5-master', 'custom', path='C:\\Users\\yaman\\Desktop\\a1b2\\alfa.pt', source='local') //if you download yolov5-master you can use this line
model = torch.hub.load('ultralytics/yolov5', 'custom', path='alfa.pt', force_reload=True)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

reader = easyocr.Reader(['en'], gpu=True)

cap = VideoStream(src=0).start()


#last_frame_time = datetime.datetime.now()
bulunan = 0
while True:
    frame = cap.read()
    if frame is None:
        continue
    #frame = cv2.resize(frame, (1500, 780))
    frame = cv2.resize(frame, (900, 780))
    result = model(frame)
    boxes = result.xyxy[0].cpu().numpy()
    overlay=frame.copy()
    alpha = 0.7
    


    for box in boxes:
        x1, y1, x2, y2, conf, class_idx = box
       
        if conf > 0.1:
     
            crop_img = frame[int(y1):int(y2), int(x1):int(x2)]
            result = reader.readtext(crop_img)
            crop_img2 = frame[int(y1)-150:int(y2)+30, int(x1)-150:int(x2)+150]

            if result:
                text= result[0][1]
                text2 = str(text)
                text2=text2.upper()
                text2 = text2.split(" ")
                text3=""
                for i in text2:
                    if i==" ":
                        pass
                    else:
                        text3+=i
                    
                
                asiye = text3.find("the plate you want")
               
                if asiye> 0 or asiye == 0:
                    cv2.rectangle(overlay, (int(x1)-150, int(y1)-150), (int(x2)+150, int(y2)+40), (0, 0, 255), -1)
                    cv2.putText(overlay,text3, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),1,cv2.LINE_AA)
                    sed = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                    frame=sed
                    #cv2.imshow("plaka",crop_img)
                    
                    if bulunan == 0:
                        img = Image.fromarray(crop_img)
                        img.save('plaka.png')
                        #img2 = cv2.imread("plaka.png")
                        #cv2.imwrite('resim.png', img)
                        cv2.imshow("plaka",crop_img2)          
                        bulunan = 1
                        
                else:
                    cv2.putText(frame,text3,(int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
    

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()