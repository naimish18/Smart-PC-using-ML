import cv2
import HandTrack as ht

cap=cv2.VideoCapture(0)
detector=ht.detect_hand(detection_con=0.8)

while True:
    success,img=cap.read()
    # count hands in web camera
    img,hands=detector.find_hands(img,True)
    # if hands found
    if hands:
        fingers_count = 0
        # iterating all hands
        for i in range(hands):
            # find location of all 21 landmarks
            ldm_list,box=detector.find_position(img,hand_no=i,draw=False)
            if ldm_list:
                # find how many fingers up
                fingers=detector.fingers_up()
                fingers_count+=sum(fingers)
        cv2.putText(img,"Fingers: "+str(fingers_count),(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    cv2.imshow("",img)
    cv2.waitKey(1)