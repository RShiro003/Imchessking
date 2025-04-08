import numpy as np
import cv2

# 캘리브레이션 결과 불러오기
mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist = np.array([distortion_coefficients], dtype=np.float32)  # distortion_coefficients는 실제 값으로 채워야 합니다.

# 왜곡 보정할 이미지 파일 경로
image_path = 'data/chessboard_undistort.jpg'  # 왜곡이 있는 이미지

img = cv2.imread(image_path)
h, w = img.shape[:2]

# 왜곡 보정 행렬 계산
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# 왜곡 보정 수행
undistorted_img = cv2.undistort(img, mtx, dist, None, new_mtx)

# 결과 이미지 출력
cv2.imshow('Original Image', img)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
