import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

#TODO 세 백터 사이의 각을 구하는 함수
def three_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0 :
        angle = 360 - angle
    return angle

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Lock을 걸고
        image.flags.writeable = False
        result = pose.process(image)
        # Lock을 풀고
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, result.pose_landmarks, 
                                mp_pose.POSE_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                                )
        

        try:
            landmarks = result.pose_landmarks.landmark

            shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z])
            
            elbow = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z])
            
            wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z])

            vector_A = elbow - shoulder
            vector_B = wrist - elbow

            # 벡터의 내적 계산
            dot_product = np.dot(vector_A, vector_B)
            # 두 벡터의 크기 계산
            norm_A = np.linalg.norm(vector_A)
            norm_B = np.linalg.norm(vector_B)
            
            # 각도 계산
            angle = np.arccos(dot_product / (norm_A * norm_B))
            # 라디안 값을 도로 변환
            angle = np.degrees(angle)
            # print(f"Elbow Angle: {angle:.2f} degrees")
            # 함수 사용
            print (three_angle(shoulder , elbow , wrist))

        except Exception as e:
            print("Error occurred:", e)

        cv2.imshow('pose', image)

        if cv2.waitKey(10) & 0XFF == ord('q'):
            break