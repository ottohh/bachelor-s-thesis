import cv2
import numpy as np
from reportlab.pdfgen import canvas
from create_marker import saveMarkerToPdf
# Parameters for the ChArUco board
ARUCO_DICT = cv2.aruco.DICT_4X4_50
SQUARES_VERTICALLY = 5
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
OUTPUT_FILE = "charuco_board3.pdf"


# Create a window to display the ChArUco board for reference
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)

# Get board image
board_size = (
    (SQUARES_HORIZONTALLY * SQUARE_LENGTH*1000),  # Width in mm
    (SQUARES_VERTICALLY * SQUARE_LENGTH*1000)    # Height in mm
)

dpi = 300 # This should be the printers dpi so we have correct amount of pixels for the Image

board_image = cv2.aruco.CharucoBoard.generateImage(board, (int(board_size[0]/25.4)*dpi, int(board_size[1]/25.4)*dpi))
saveMarkerToPdf(board_image,OUTPUT_FILE,board_size)

# Prepare to capture calibration images
cap = cv2.VideoCapture(0)  # Replace 0 with your camera index
all_corners = []
all_ids = []
image_size = None

print("Press 'c' to capture images for calibration or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray,dictionary)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

    cv2.imshow("Calibration", frame)
    key = cv2.waitKey(1)
    if key == ord("c") and  ids is not None and len(ids)>4 and charuco_corners is not None and  len(charuco_corners)>4:
        print("Captured frame for calibration.")
        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)
        if image_size is None:
            image_size = gray.shape[::-1]
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if len(all_corners) < 5:
    print("Not enough images for calibration. Capture at least 5.")
    exit()

# Camera calibration
print("Calibrating camera...")
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_corners,
    charucoIds=all_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None,
)

if ret:
    print("Calibration successful!")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
else:
    print("Calibration failed.")

# Save calibration data
np.savez("camera_calibration_data.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print("Calibration data saved as 'camera_calibration_data.npz'.")