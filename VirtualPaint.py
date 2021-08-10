import cv2
import HandTrack as ht
import numpy as np

cap=cv2.VideoCapture(0)
width=1280
height=720
cap.set(3,width)
cap.set(4,height)
image=cv2.imread('HeaderPaint.jpg')
detector=ht.detect_hand(detection_con=0.8)
color=(255,0,255)
brush_size=10
x_prev,y_prev=0,0
canvas=np.zeros((720,1280,3),np.uint8)
brush_size_pos=(100,500)

while True:
    success,img=cap.read()
    img=cv2.flip(img,1)
    img,hands=detector.find_hands(img)
    ldm_list,box=detector.find_position(img,draw=False)
    if ldm_list:
        i_x,i_y=ldm_list[8][1],ldm_list[8][2]
        m_x,m_y=ldm_list[12][1],ldm_list[12][2]

        fingers=detector.fingers_up()
        if fingers[1] and fingers[2] and fingers[3]:
            if 70<m_x<130 and 300<m_y<500:
                if 500>m_y>490:
                    brush_size=10
                    brush_size_pos=(100,500)
                if 490>m_y>470:
                    brush_size=20
                    brush_size_pos=(100,480)
                if 470>m_y>450:
                    brush_size=30
                    brush_size_pos=(100,460)
                if 450>m_y>430:
                    brush_size=40
                    brush_size_pos=(100,440)
                if 430>m_y>410:
                    brush_size=50
                    brush_size_pos=(100,420)
                if 410>m_y>390:
                    brush_size=60
                    brush_size_pos=(100,400)
                if 390>m_y>370:
                    brush_size=70
                    brush_size_pos=(100,380)
                if 370>m_y>350:
                    brush_size=80
                    brush_size_pos=(100,360)
                if 350>m_y>330:
                    brush_size=90
                    brush_size_pos=(100,340)
                if 330>m_y>310:
                    brush_size=100
                    brush_size_pos=(100,320)

        if fingers[1] and fingers[2]:
            x_prev,y_prev=0,0
            cv2.rectangle(img,(i_x,i_y-20),(m_x,m_y+20),color,cv2.FILLED)
            if i_y<150:
                if 370<i_x<470:
                    color=(255,0,255)
                if 570<i_x<670:
                    color=(0,0,255)
                if 770<i_x<870:
                    color=(0,255,0)
                if 1000<i_x<1280:
                    color=(0,0,0)

        if fingers[1] and not fingers[2]:
            cv2.circle(img,(i_x,i_y),brush_size,color,cv2.FILLED)

            if x_prev==0 and y_prev==0:
                x_prev,y_prev=i_x,i_y
            cv2.line(img,(x_prev,y_prev),(i_x,i_y),color,brush_size)
            cv2.line(canvas, (x_prev, y_prev), (i_x, i_y), color, brush_size)
            x_prev,y_prev=i_x,i_y

    gray_img=cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
    _,inv_img=cv2.threshold(gray_img,50,255,cv2.THRESH_BINARY_INV)
    inv_img=cv2.cvtColor(inv_img,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,inv_img)
    img=cv2.bitwise_or(img,canvas)

    cv2.line(img,(100,500),(100,300),(255,0,0),10)
    cv2.putText(img,"Size: "+str(brush_size),(50,250),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.circle(img,brush_size_pos,10,(255,255,255),cv2.FILLED)
    img[0:150,0:1280]=image
    cv2.imshow("img",img)
    cv2.waitKey(1)