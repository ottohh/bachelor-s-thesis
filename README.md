# bachelor-s-thesis
This repository contains code used in my bachelor's thesis to compare the localization accuracy of ArUco and AprilTag fiducial markers.

## Files in the repository
process_results.py contains the code that detects markers from the recorded video. It calculates the errors in distances compared to ground truth data from Motion Capture Studio.

readCameraAndTagCoordinates.py reads the camera and tag coordinates at the specific frame of the video. It is used in process_results.py.

detect_frame_change.py was used to determine when the camera started to move to sync the mocap recording with the video.

calculate_coordinates_aruco.py was used as a demo in the seminar to show people what pose estimation does. The cube follows the camera movement.

create_marker.py and calibrate_camera.py were used for marker creation and camera calibration.
