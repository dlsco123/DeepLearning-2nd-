import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

model = joblib.load('mediapipe/face.pkl')
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

cname = ['happy', 'sad']  # 표정 카테고리를 전역 변수로 설정
font = cv2.FONT_HERSHEY_COMPLEX

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 영상을 캡쳐하지 못했습니다.")
            continue
        frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(image_rgb)

        if result.face_landmarks:
            mp_drawing.draw_landmarks(frame, result.face_landmarks,
                                    mp_holistic.FACEMESH_CONTOURS,
                                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=1))

            face = result.face_landmarks.landmark
            face_row = [coord for landmark in face for coord in (landmark.x, landmark.y, landmark.z)]

            yhat = model.predict(pd.DataFrame([face_row]))[0]
            cv2.putText(frame, cname[yhat], (30, 30), font, 1, (255, 0, 0), 2)

        cv2.imshow('face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
