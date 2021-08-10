import cv2
import numpy as np
import HandTrack as ht
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap=cv2.VideoCapture(0)
detector=ht.detect_hand(detection_con=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

max_vol=volume.GetVolumeRange()[1]
min_vol=volume.GetVolumeRange()[0]

area=0

while True:
    success,img=cap.read()
    img,hands=detector.find_hands(img)
    ldm_list,box=detector.find_position(img,draw=True)
    if ldm_list:

        area=((box[2]-box[0])*(box[3]-box[1]))//100

        if 250<area<1000:
            img,length,info=detector.find_distance(4,8,img,True)
            vol_height=np.interp(length,[50,200],[370,120])
            vol=np.interp(length,[50,200],[0,100])

            smooth = 10
            vol = smooth * round(vol / smooth)

            finger_up=detector.fingers_up()[4]
            if not finger_up:
                volume.SetMasterVolumeLevelScalar(vol/100, None)
                cv2.circle(img,(info[4],info[5]),10,(0,0,255),cv2.FILLED)

            cv2.rectangle(img,(20,120),(55,370),(0,0,255),3)
            print(vol_height)
            cv2.rectangle(img, (20, int(vol_height)), (55, 370), (0, 0, 255), cv2.FILLED)
            cv2.putText(img,str(vol)+'%',(20,400),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),3)

    cv2.imshow("",img)
    cv2.waitKey(1)