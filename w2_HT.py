import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)


mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
hands=mp_hands.Hands(min_tracking_confidence=0.3)


while True:
    sucess,frame = video.read()
    # frame = cv2.resize(frame,(1200,720))
    results=hands.process(frame)

    if results.multi_hand_landmarks:

        for hand_landmarks in (results.multi_hand_landmarks):

            mp_drawing.draw_landmarks(image=frame,landmark_list=hand_landmarks,connections=mp_hands.HAND_CONNECTIONS)


    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
video.release
cv2.destroyAllWindows()