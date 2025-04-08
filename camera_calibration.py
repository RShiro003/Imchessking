import numpy as np
import cv2
import glob

# 체스보드 패턴 크기
chessboard_size = (9, 6)  # 체스보드의 내부 코너 수 (가로 9, 세로 6)

# 3D 세계 좌표 준비 (0,0,0 부터 시작하는 체스보드 코너 좌표)
object_points = []
for i in range(chessboard_size[1]):
    for j in range(chessboard_size[0]):
        object_points.append([j, i, 0])

object_points = np.array(object_points, dtype=np.float32)
object_points = object_points.reshape(-1, 3)

# 저장할 변수들
object_points_all = []  # 3D 좌표
image_points_all = []  # 2D 이미지 좌표

# 영상 파일 경로 (체스보드가 찍힌 이미지 파일들)
images = glob.glob('C:\iCloudDrive\Seoultceh\3\CV\homework\Week04\Imchessking\chessboard1.jpg')  # 자신의 이미지 경로로 수정

for image_file in images:
    img = cv2.imread(image_file)

    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size)

    if ret:
        image_points_all.append(corners)
        object_points_all.append(object_points)

        # 코너 위치 그리기
        img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard corners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

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
