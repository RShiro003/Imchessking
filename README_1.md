# Camera Calibration Results

## Overview
This document presents the results of the **camera calibration** process using a chessboard pattern. The calibration was performed by capturing a video of the chessboard from various angles and applying camera calibration techniques to calculate the intrinsic parameters of the camera.

## Calibration Data
The camera calibration parameters used for distortion correction are:

### Camera Matrix (Intrinsic Parameters):
```python
mtx = np.array([[823.37, 0, 1005.50],
                [0, 835.81, 545.83],
                [0, 0, 1]], dtype=np.float32)
