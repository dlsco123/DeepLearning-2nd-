import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
# import socket

# 변수 및 모델 초기화
max_num_hands = 1
gesture = {0: 'stop', 1: 'fire'}

# 데이터 로드 및 KNN 학습
df = pd.read_csv('hands.csv', header=None)
x = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(x, cv2.ml.ROW_SAMPLE, y)

# Mediapipe 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
total_result = []

# 클릭 이벤트 핸들러
def click(event, x, y, flags, params):
    data = params['data']
    if event == cv2.EVENT_LBUTTONDOWN:
        print('mouse click')
        total_result.append(data)
        print(data)


# 이벤트 핸들러에 data 전달
params = {'data': None}  # 초기화
cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', click, param=params)


with mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for res in result.multi_hand_landmarks:
                joint = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark])
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                
                # 각도 계산
                angle = np.degrees(np.arccos(np.einsum('nt,nt->n', v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :], v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :])))
                data = angle.astype(np.float32)
                params['data'] = data
                ret, rdata, neig, dist = knn.findNearest([data], 5)
                idx = int(rdata[0][0])
                print(idx)
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Dataset', img)
        if cv2.waitKey(1) == ord('q'):
            break

total_result = np.array(total_result, dtype=np.float32)
df = pd.DataFrame(total_result)
df.to_csv('hands.csv', mode='a', index=False, header=False)
print('=========end==============')
