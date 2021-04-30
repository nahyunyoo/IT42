import sys
import numpy as np
import cv2

cam = cv2.VideoCapture(0)

if not cam.isOpened():
	print('Camera open failed!')
	sys.exit()

w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) #기본 카메라 넓이
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) #기본 카메라 높이


# 객체 생성 및 학습 데이터 불러오기
face_classifier = cv2.CascadeClassifier('.\haarcascade_frontalface_alt2.xml')#얼굴 검출
eye_classifier = cv2.CascadeClassifier('.\haarcascade_eye.xml')#눈 검출

if face_classifier.empty() or eye_classifier.empty():
    print('XML file load failed')
    sys.exit()

while True:
    #비디오의 한 프레임씩 읽는다.
	#제대로 읽으면 ret는 True, frame에는 읽은 프레임이 나온다.
    ret, frame = cam.read()

    if not ret:
        break
    
    # 멀티 스케일 객체 검출 함수
    faces = face_classifier.detectMultiScale(frame, scaleFactor=1.2,
											 minSize=(100, 100), maxSize=(400, 400))

    # 영상 받아오기
    for (x, y, w, h) in faces:
        # 얼굴 빨간색 사각형 그리기
        cv2.rectangle(frame, (x, y, w, h), (255, 0, 255), 2) 

        #눈 검출
        face_half = frame[y:y + h // 2, x:x + w] #위 화면에서만 (빨리 찾기 위함)
        eyes = eye_classifier.detectMultiScale(face_half)

        for (ex, ey, ew, eh) in eyes:
            # 눈 파란색 사각형 그리기
            cv2.rectangle(face_half, (ex, ey, ew, eh), (255, 0, 0), 2) 

    cv2.imshow('IT42', frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()