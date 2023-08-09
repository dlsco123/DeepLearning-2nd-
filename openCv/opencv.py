# !pip install opencv-python

import cv2

# img=cv2.imread('image/starry_night.jpg')

# cv2.imshow('img', img)
# cv2.waitKey()

# cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # 좌우 반전
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

