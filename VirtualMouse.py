import mouse
import cv2
import HandTrack as ht
import ctypes
import numpy as np

cap=cv2.VideoCapture(0)
detector=ht.detect_hand(maxHands=1,detection_con=0.7)
width_cam,height_cam=640,480
width_screen,height_screen=ctypes.windll.user32.GetSystemMetrics(0),ctypes.windll.user32.GetSystemMetrics(1)
prev_x,prev_y=mouse.get_position()
size_reduction=100

while True:
    success,img=cap.read()
    img=cv2.flip(img,1)
    img,hands=detector.find_hands(img)
    ldm_list,box=detector.find_position(img)
    if ldm_list:
        i_x,i_y=ldm_list[4][1],ldm_list[4][2]
        m_x,m_y=ldm_list[8][1],ldm_list[8][2]

        fingers=detector.fingers_up()
        cv2.rectangle(img,(size_reduction,0),(width_cam-size_reduction,height_cam-2*size_reduction),(0,0,255),2)
        if fingers[1] and not fingers[2]:
            x=np.interp(i_x,(size_reduction,width_cam-size_reduction),(0,width_screen))
            y=np.interp(i_y,(0,height_cam-2*size_reduction),(0,height_screen))

            mouse.move(x-prev_x,y-prev_y,absolute=False)
            prev_x,prev_y=x,y

        if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            mouse.click('left')

    else:
        prev_x, prev_y = mouse.get_position()


    cv2.imshow("",img)
    cv2.waitKey(1)


