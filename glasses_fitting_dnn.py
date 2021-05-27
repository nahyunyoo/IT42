import sys
import numpy as np
import cv2

#3채널 img 영상에 4채널 item 영상을 pos위치에 합성
def overlay(img, glasses, pos):
	#실제 합성을 수행할 부분 영상 좌표 계산
	sx = pos[0]
	ex = pos[0] + glasses.shape[1]
	sy = pos[1]
	ey = pos[1] + glasses.shape[0]

	if sx < 0 or sy < 0 or ex > img.shape[1] or ey > img.shape[0]:
		return

	#부분 영상 참조. 
	# img1: 입력 영상의 부분 영상
	# img2: 안경 영상의 부분 영상
	img1 = img[sy:ey, sx:ex] #shape=(h, w, 3)
	img2 = glasses[:, :, 0:3] #shape=(h, w, 3)
	alpha = 1. - (glasses[:, :, 3] / 255.) #shape=(h, w)

	# BGR 채널별로 두 부분 영상의 가중합
	img1[..., 0] = (img1[..., 0] * alpha + img2[..., 0] * (1. - alpha))
	img1[..., 1] = (img1[..., 1] * alpha + img2[..., 1] * (1. - alpha))
	img1[..., 2] = (img1[..., 2] * alpha + img2[..., 2] * (1. - alpha))



model = '.\opencv_face_detector\\res10_300x300_ssd_iter_140000_fp16.caffemodel'
config = '.\opencv_face_detector\\deploy.prototxt'
#model = 'opencv_face_detector/opencv_face_detector_uint8.pb'
#config = 'opencv_face_detector/opencv_face_detector.pbtxt'

cap = cv2.VideoCapture(0)

if not cap.isOpened():
	print('Camera open failed!')
	sys.exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #기본 카메라 넓이
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #기본 카메라 높이

# 합성된 결과 동영상 파일로 저장
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video_out = cv2.VideoWriter('output.avi', fourcc, 10, (w, h), isColor=True)

eye_classifier = cv2.CascadeClassifier('./haarcascade_eye.xml')#눈 검출
net = cv2.dnn.readNet(model, config)

if not video_out.isOpened():
	print('video open failed')
	sys.exit()

if eye_classifier.empty():
	print('XML load failed!')
	sys.exit()

if net.empty():
	print('Net open failed!')
	sys.exit()


# 안경 PNG 파일 열기 (Image from http://www.pngall.com/)
# 알파채널: 기본 채널외에 사용자가 선택영역을 저장하고 다시 불러(LOAD)사용할 수 있는 것으로 선택부분은 흰색으로 
# 선택되지 않는 부분은 검정색으로 표현한다.
glasses = cv2.imread('.\glasses\eyes-2022424_1280.png', cv2.IMREAD_UNCHANGED) 

if glasses is None:
	print('PNG image open failed!')
	sys.exit()

# 합성 할 안경 영상 위치좌표 계산
ew, eh = glasses.shape[:2]  # 가로, 세로 크기
ex1, ey1 = 240, 220  # 왼쪽 눈 좌표
ex2, ey2 = 660, 220  # 오른쪽 눈 좌표


while True:
	ret, frame = cap.read()

	if not ret:
		break

	blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
	net.setInput(blob)
	out = net.forward()

	detect = out[0, 0, :, :]
	(h, w) = frame.shape[:2]

	for i in range(detect.shape[0]):
		confidence = detect[i, 2]
		if confidence < 0.5:
			break

		face_x1 = int(detect[i, 3] * w)
		face_y1 = int(detect[i, 4] * h)
		face_x2 = int(detect[i, 5] * w)
		face_y2 = int(detect[i, 6] * h)

		cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 0))

		label = f'Face: {confidence:4.2f}'
		cv2.putText(frame, label, (face_x1, face_y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

		#눈 검출
		faceROI = frame[face_y1: face_y1 + (face_y2 - face_y1) // 2, face_x1:face_x1 + (face_x2 - face_x1)]
		eyes = eye_classifier.detectMultiScale(faceROI)

		if len(eyes) != 2:
			continue

		# 두 개의 눈 중앙 위치를 (x1, y1), (x2, y2) 좌표로 저장
		eye_x1 = face_x1 + eyes[0][0] + (eyes[0][2] // 2)
		eye_y1 = face_y1 + eyes[0][1] + (eyes[0][3] // 2)
		eye_x2 = face_x1 + eyes[1][0] + (eyes[1][2] // 2)
		eye_y2 = face_y1 + eyes[1][1] + (eyes[1][3] // 2)

		if eye_x1 > eye_x2:
			eye_x1, eye_y1, eye_x2, eye_y2 = eye_x2, eye_y2, eye_x1, eye_y1

		# 두 눈 사이의 거리를 이용하여 스케일링 팩터를 계산 (두 눈이 수평하다고 가정)
		# (x2-x1): 실제 입력영상에서의 두 눈의 간격
		# (ex2 - ex1): PNG파일(안경영상)에서의 두 눈의 간격
		fx = (eye_x2 - eye_x1) / (ex2 - ex1)
		glasses2 = cv2.resize(glasses, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_AREA)

		# 크기 조절된 안경 영상을 합성할 위치 계산 (좌상단 좌표)
		# resize한 png영상이 실제 카메라 프레임 어느 위치에서 시작되서 할것인가(좌측상단 좌표)
		pos = (eye_x1 - int(ex1 * fx), eye_y1 - int(ey1 * fx))

		# 영상 합성
		overlay(frame, glasses2, pos)

	# 프레임 저장 및 화면 출력
	cv2.imshow('frame', frame)
	video_out.write(frame)

	if cv2.waitKey(1) == 27:
		break

cap.release()
video_out.release()
cv2.destroyAllWindows()
