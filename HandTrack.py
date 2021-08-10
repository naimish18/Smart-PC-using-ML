import cv2
import mediapipe as mp
import math

class detect_hand():
    def __init__(self,mode=False,maxHands=2,detection_con=0.5,track_con=0.5):
        # initialize the parameters for hand module
        self.mode=mode
        self.maxHands=maxHands
        self.detection_con=detection_con
        self.track_con=track_con

        self.hands_mp=mp.solutions.hands
        self.hands=self.hands_mp.Hands(self.mode,self.maxHands,self.detection_con,self.track_con)
        self.mp_draw=mp.solutions.drawing_utils

    def find_hands(self,img,show=True):
        # covert BGR image to RGB ( because hand module require RGB image )
        rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # process RGB image
        self.res=self.hands.process(rgb_img)
        # to find the landmarks of hand
        if self.res.multi_hand_landmarks:
            # iterating over all 21 landmarks of hand
            for hand_ldm in self.res.multi_hand_landmarks:
                if show:
                    # draw all landmarks on image
                    self.mp_draw.draw_landmarks(img,hand_ldm,self.hands_mp.HAND_CONNECTIONS,)
        if self.res.multi_hand_landmarks:
            hands=len(self.res.multi_hand_landmarks)
        else:
            hands=0
        return img,hands

    def find_position(self,img,hand_no=0,draw=True,pos=0):
        self.ldm_list=[]
        x_list=[]
        y_list=[]
        box=[]
        if self.res.multi_hand_landmarks:
            # if there are multiple hands detected then for which hand you want to find landmarks
            my_hand=self.res.multi_hand_landmarks[hand_no]
            for id, ldm in enumerate(my_hand.landmark):
                # to get height and width picture
                h, w, c = img.shape
                # find landmarks x and y according to height
                cx, cy = int(ldm.x * w), int(ldm.y * h)
                self.ldm_list.append([id,cx,cy,ldm.z])
                x_list.append(cx)
                y_list.append(cy)
            box=min(x_list),min(y_list),max(x_list),max(y_list)
            if draw:
                cv2.rectangle(img,(box[0]-15,box[1]-15),(box[2]+15,box[3]+15),(0,255,0),2)

        return self.ldm_list,box

    def fingers_up(self):
        fingers=[]
        hand=0
        # detect is it left or right hand
        if self.ldm_list[5][1]<self.ldm_list[9][1]<self.ldm_list[13][1]<self.ldm_list[17][1]:
            hand=1
        # check how many fingers up
        if hand:
            if self.ldm_list[4][1]<self.ldm_list[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if self.ldm_list[4][1] > self.ldm_list[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        for i in range(2,6):
            if self.ldm_list[i*4][2]<self.ldm_list[i*4-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def find_distance(self,landmark_1,landmark_2,img,draw=True):
        x1, y1 = self.ldm_list[landmark_1][1], self.ldm_list[landmark_1][2]
        x2, y2 = self.ldm_list[landmark_2][1], self.ldm_list[landmark_2][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2

        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        # distance between two landmarks
        length = math.sqrt(pow(abs(x1 - x2), 2) + pow(abs(y1 - y2), 2))
        info=[x1,x2,y1,y2,cx,cy]
        return img,length,info
