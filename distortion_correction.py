import numpy as np
import cv2

# 카메라 매트릭스와 왜곡 계수 설정 (예시 값: 실제 캘리브레이션 결과로 바꿔주세요)
mtx = np.array([[823.37, 0, 1005.50],
                [0, 835.81, 545.83],
                [0, 0, 1]], dtype=np.float32)

dist = np.array([0.07477567, 0.33346671, -0.00245167, 0.00885194, -1.09295097])

# 동영상 파일 경로 (자신의 동영상 경로로 수정)
video_path = r'C:\Project\Imchessking\IMG_4213.avi'  # 동영상 파일 경로

# 동영상 파일 열기
cap = cv2.VideoCapture(video_path)

# 동영상이 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("동영상을 열 수 없습니다. 경로를 확인하세요.")
    exit()

frame_count = 0
frame_skip = 5  # 5번째 프레임마다 처리 (속도 개선)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 동영상의 회전 처리 (예: 90도 회전)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 시계방향 90도 회전

    # 매 프레임마다 처리하지 않고, 특정 프레임만 처리
    if frame_count % frame_skip == 0:
        # 동영상의 크기 얻기
        h, w = frame.shape[:2]

        # 왜곡 보정된 카메라 매트릭스 계산
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # 왜곡 보정된 이미지
        undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_mtx)

        # 왜곡 보정된 이미지와 원본 이미지 출력
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Undistorted Frame", undistorted_frame)

        cv2.waitKey(1)  # 1ms 동안 기다리며 출력

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
