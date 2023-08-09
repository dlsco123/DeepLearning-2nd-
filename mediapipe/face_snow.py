import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

faceimg = cv2.imread('face_mk.png', cv2.IMREAD_UNCHANGED)

def change_mask(background_img, img_to_overlay, x, y, overlay_size=None):
    try:
        bg_img = background_img.copy()

        if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

        if overlay_size is not None:
            img_to_overlay = cv2.resize(img_to_overlay.copy(), overlay_size)

        b, g, r, a = cv2.split(img_to_overlay)
        mask = cv2.medianBlur(a, 5)

        h, w, _ = img_to_overlay.shape
        roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(img_to_overlay, img_to_overlay, mask=mask)

        bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = img1_bg + img2_fg
        return cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    except Exception as e:
        print(e)
        return background_img

while True:
    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceMesh.process(imgRGB)
    
    if result.multi_face_landmarks:
        for faceLms in result.multi_face_landmarks:
            xy_point = []
            for c, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                xy_point.append([lm.x, lm.y])

            mean_xy = np.mean(xy_point, axis=0)
            face_width = faceimg.shape[1]
            face_height = faceimg.shape[0]

            img = change_mask(img, faceimg, int(mean_xy[0]*iw), int(mean_xy[1]*ih), (face_width, face_height))
    else:
        result = img

    cv2.imshow('image', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()