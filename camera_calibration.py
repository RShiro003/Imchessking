import numpy as np
import cv2

# 체스보드 크기 (내부 코너 수)
chessboard_size = (9, 6)

# 3D 좌표 준비
object_points = []
for i in range(chessboard_size[1]):
    for j in range(chessboard_size[0]):
        object_points.append([j, i, 0])

object_points = np.array(object_points, dtype=np.float32)
object_points = object_points.reshape(-1, 3)

# 저장할 변수들
object_points_all = []  # 3D 좌표
image_points_all = []   # 2D 이미지 좌표

# 동영상 파일 경로 (수정된 경로)
video_path = r'C:\Project\Imchessking\IMG_4213.avi'  # 원시 문자열로 수정

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

    # 매 프레임마다 처리하지 않고, 특정 프레임만 처리
    if frame_count % frame_skip == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환

        # 체스보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size)

        if ret:
            image_points_all.append(corners)
            object_points_all.append(object_points)

            # 코너 위치 그리기
            frame = cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
            cv2.imshow('Chessboard corners', frame)  # 체스보드 코너 그린 영상 출력
            cv2.waitKey(500)  # 500ms 동안 기다리며 보여주기

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# 체스보드 이미지가 하나라도 검출되었는지 확인
if len(object_points_all) == 0:
    print("체스보드 코너를 찾을 수 없습니다. 동영상을 확인하고 다시 시도하십시오.")
    exit()

# 카메라 캘리브레이션 수행
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points_all, image_points_all, gray.shape[::-1], None, None)

# 캘리브레이션 결과 출력
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# 카메라 캘리브레이션 파라미터 (fx, fy, cx, cy)
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]

# RMSE 계산
total_error = 0
for i in range(len(object_points_all)):
    projected_points, _ = cv2.projectPoints(object_points_all[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(image_points_all[i], projected_points, cv2.NORM_L2) / len(projected_points)
    total_error += error

mean_error = total_error / len(object_points_all)
print("Reprojection error (RMSE):", mean_error)
