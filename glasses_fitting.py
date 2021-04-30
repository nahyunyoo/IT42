# 스노우 앱 
# 카메라 입력 영상에서 얼굴을 검출하여 재미있는 그래픽을 합성하는 프로그램 

# 구현할 기능 
# 카메라 입력 영상에서 얼굴&눈 검출하기
# - 캐스케이드 분류기 사용 
# 눈 위치와 맞게 투명한 PNG 파일 합성하기
# - PNG 영상 크기 조절 및 위치 계산  
# 합성된 결과를 동영상으로 저장하기 

import sys #python interpreter: python의 소스코드를 바로 실행하는 컴퓨터 프로그램 또는 환경
import numpy as np
import cv2


# 3채널 img 영상에 4채널 item 영상을 pos 위치에 합성
def overlay(img, glasses, pos):
	# 실제 합성을 수행할 부분 영상 좌표 계산
	sx = pos[0]
	ex = pos[0] + glasses.shape[1]
	sy = pos[1]
	ey = pos[1] + glasses.shape[0]

	# 합성할 영역이 입력 영상 크기를 벗어나면 무시
	if sx < 0 or sy < 0 or ex > img.shape[1] or ey > img.shape[0]:
		return

	# 부분 영상 참조. img1: 입력 영상의 부분 영상, img2: 안경 영상의 부분 영상
	img1 = img[sy:ey, sx:ex]   # shape=(h, w, 3)
	img2 = glasses[:, :, 0:3]  # shape=(h, w, 3)
	alpha = 1. - (glasses[:, :, 3] / 255.)  # shape=(h, w)

	# BGR 채널별로 두 부분 영상의 가중합
	img1[..., 0] = (img1[..., 0] * alpha + img2[..., 0] * (1. - alpha)).astype(np.uint8)
	img1[..., 1] = (img1[..., 1] * alpha + img2[..., 1] * (1. - alpha)).astype(np.uint8)
	img1[..., 2] = (img1[..., 2] * alpha + img2[..., 2] * (1. - alpha)).astype(np.uint8)



# 카메라 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
	print('Camera open failed!')
	sys.exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #기본 카메라 넓이
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #기본 카메라 높이

# 합성된 결과 동영상 파일로 저장
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 30, (w, h))

# Haar-like XML 파일 열기
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')#얼굴 검출
eye_classifier = cv2.CascadeClassifier('./haarcascade_eye.xml')#눈 검출

if face_classifier.empty() or eye_classifier.empty():
	print('XML load failed!')
	sys.exit()

# 안경 PNG 파일 열기 (Image from http://www.pngall.com/)
# 알파채널: 기본 채널외에 사용자가 선택영역을 저장하고 다시 불러(LOAD)사용할 수 있는 것으로 선택부분은 흰색으로 
# 선택되지 않는 부분은 검정색으로 표현한다.
glasses = cv2.imread('.\glasses.png', cv2.IMREAD_UNCHANGED) 

if glasses is None:
	print('PNG image open failed!')
	sys.exit()

# 합성 할 안경 영상 위치좌표 계산
ew, eh = glasses.shape[:2]  # 가로, 세로 크기
ex1, ey1 = 240, 300  # 왼쪽 눈 좌표
ex2, ey2 = 660, 300  # 오른쪽 눈 좌표

# 매 프레임에 대해 얼굴 검출 및 안경 합성
while True:
	#비디오의 한 프레임씩 읽는다.
	#제대로 읽으면 ret는 True, frame에는 읽은 프레임이 나온다.
	ret, frame = cap.read()

	if not ret:
		break

	# 얼굴 검출
	faces = face_classifier.detectMultiScale(frame, scaleFactor=1.2,
											 minSize=(100, 100), maxSize=(400, 400))

	# 얼굴 검출한 부분에서 눈 검출하기
	for (x, y, w, h) in faces:
		#cv2.rectangle(frame, (x, y, w, h), (255, 0, 255), 2)

		# 눈 검출
		faceROI = frame[y:y + h // 2, x:x + w]
		eyes = eye_classifier.detectMultiScale(faceROI)

		# 눈을 2개 검출한 것이 아니라면 무시
		if len(eyes) != 2:
			continue

		# 두 개의 눈 중앙 위치를 (x1, y1), (x2, y2) 좌표로 저장
		x1 = x + eyes[0][0] + (eyes[0][2] // 2)
		y1 = y + eyes[0][1] + (eyes[0][3] // 2)
		x2 = x + eyes[1][0] + (eyes[1][2] // 2)
		y2 = y + eyes[1][1] + (eyes[1][3] // 2)

		if x1 > x2:
			x1, y1, x2, y2 = x2, y2, x1, y1

		#cv2.circle(faceROI, (x1, y1), 5, (255, 0, 0), 2, cv2.LINE_AA)
		#cv2.circle(faceROI, (x2, y2), 5, (255, 0, 0), 2, cv2.LINE_AA)

		# 두 눈 사이의 거리를 이용하여 스케일링 팩터를 계산 (두 눈이 수평하다고 가정)
		# (x2-x1): 실제 입력영상에서의 두 눈의 간격
		# (ex2 - ex1): PNG파일(안경영상)에서의 두 눈의 간격
		fx = (x2 - x1) / (ex2 - ex1)
		glasses2 = cv2.resize(glasses, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_AREA)

		# 크기 조절된 안경 영상을 합성할 위치 계산 (좌상단 좌표)
		# resize한 png영상이 실제 카메라 프레임 어느 위치에서 시작되서 할것인가(좌측상단 좌표)
		pos = (x1 - int(ex1 * fx), y1 - int(ey1 * fx))

		# 영상 합성
		overlay(frame, glasses2, pos)

	# 프레임 저장 및 화면 출력
	out.write(frame)
	cv2.imshow('frame', frame)

	if cv2.waitKey(1) == 27:
		break

cap.release()
out.release()
cv2.destroyAllWindows()
