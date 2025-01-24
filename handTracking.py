#Import all the Required libraries
import cv2
import mediapipe as mp
import time
#Create a Video Capture Object
cap = cv2.VideoCapture(0)

#Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()

#Draw the Hand landmarks
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
while True:
    ret, frame = cap.read()
    if ret:
        #Convert the frame from BGR to RGB format
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)
        if results.multi_hand_landmarks:
            for handType, handLMS in zip(results.multi_handedness, results.multi_hand_landmarks):
                for id, lm in enumerate(handLMS.landmark):
                    h, w, c = frame.shape
                    cx, cy, cz = int(lm.x*w), int(lm.y*h), int(lm.z*w)
                    print(id, cx, cy, cz)
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
                mpDraw.draw_landmarks(frame, handLMS, mpHands.HAND_CONNECTIONS)
                if handType.classification[0].label == "Right":
                    type = "Left"
                else:
                    type = "Right"
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()