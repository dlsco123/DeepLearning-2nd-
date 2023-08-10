import cv2
import mediapipe as mp
import numpy as np
import math
import socket

#which one is the best for hands? udp / tcp == udp!

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#TODO for UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 상대방 주소, 포트번호
sendport = ('127.0.0.1', 5053)

cap = cv2.VideoCapture(0)
# 손 추출 (1개)
# with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5,
#                     min_tracking_confidence=0.5) as hands:
#     while True:
#         ret, image = cap.read()
#         # 이미지 좌우반전
#         image = cv2.flip(image, 1)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = hands.process(image)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         if results.multi_hand_landmarks:
#             for hands_landmarks in results.multi_hand_landmarks:
#                 print(hands_landmarks)
#                 print('-----------------')

#             mp_drawing.draw_landmarks(image, hands_landmarks,
#                                     mp_hands.HAND_CONNECTIONS)
        
#         cv2.imshow('hand', image)
        
#         if cv2.waitKey(1) == ord('q'):
#             break

#TODO multi 손을 인식하여 그리기 위해서 mp_drawing.draw_landmarks를 if문 안으로
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:  # 웹캠에서 프레임을 제대로 읽어오지 못했을 때
            print("웹캠에서 영상을 캡쳐하지 못했습니다.")
            continue
        # 이미지 좌우반전
        image = cv2.flip(frame, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hands_landmarks in results.multi_hand_landmarks:
                p1 = hands_landmarks.landmark[4]
                p2 = hands_landmarks.landmark[8]

                a = p1.x - p2.x
                b = p1.y -p2.y
                c = math.sqrt((a**2) + (b**2))
                vol = int(c*100)
                vol = np.abs(vol)
                senddata=str(vol)

                #TODO socket통신
                sock.sendto(str.encode(senddata),sendport)

                # f1 = np.array([p1.x,p1.y, p1.z])
                # f2 = np.array([p2.x,p2.y, p2.z])
                # print(np.linalg.norm(f2-f1))


                # print("vol ==", vol)                
                cv2.putText(image, text='Volume : %d' %vol,
                            org =(10,30), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=1, color=255,thickness=2)
                # print(hands_landmarks)
                # print('-----------------')
                mp_drawing.draw_landmarks(image, hands_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow('hand', image)
        
        if cv2.waitKey(1) == ord('q'):
            break