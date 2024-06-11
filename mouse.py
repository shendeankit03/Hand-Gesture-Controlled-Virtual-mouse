import time
import pyautogui
import cv2
import numpy as np
import mediapipe as mp
from IPython.display import Image

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0

smoothening = 9
plocx, plocy = 0, 0
clocx, clocy = 0, 0

time_between_clicks = 0.5
last_click_time = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:
                    cv2.circle(img=frame, center=(x, y), radius=15, color=(0, 255, 255))
                    index_x = (screen_width / frame_width) * x
                    index_y = (screen_height / frame_height) * y
                    clocx = plocx + (index_x - plocx) / smoothening
                    clocy = plocy + (index_y - plocy) / smoothening
                    pyautogui.moveTo(clocx, clocy)
                    plocx, plocy = clocx, clocy

                if id == 4:
                    cv2.circle(img=frame, center=(x, y), radius=15, color=(0, 255, 255))
                    thumb_x = (screen_width / frame_width) * x
                    thumb_y = (screen_height / frame_height) * y
                    distance = abs(index_y - thumb_y)

                    if distance < 70:
                        if abs(thumb_x - index_x) < 50:
                            pyautogui.click()
                            print('Left-click')
                        else:
                            current_time = time.time()
                            time_since_last_click = current_time - last_click_time

                            if time_since_last_click < time_between_clicks:
                                pyautogui.doubleClick()
                                print('Double-click')
                                last_click_time = 0
                            else:
                                pyautogui.click()
                                print('Single-click')
                                last_click_time = current_time

                    # Right-click trigger using the middle finger (point number 12)
                    elif id == 12:
                        cv2.circle(img=frame, center=(x, y), radius=15, color=(255, 0, 0))
                        middle_x = (screen_width / frame_width) * x
                        middle_y = (screen_height / frame_height) * y
                        pyautogui.click(button='right')
                        print('Right-click')

    cv2.imshow('Virtual Mouse', frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()