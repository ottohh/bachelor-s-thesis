import cv2
import numpy as np
import time




# Define the dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
april_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
# Define the marker side length in meters
MARKER_LENGTH = 0.05  # For example, 5 cm



aruco_params = cv2.aruco.DetectorParameters()


aruco_params.adaptiveThreshWinSizeMin = 5
aruco_params.adaptiveThreshWinSizeMax = 50  # Increase max size for larger markers
aruco_params.adaptiveThreshWinSizeStep = 5

aruco_params.minMarkerPerimeterRate = 0.001 # Keyboard keys are detected as markers so had to increase this.
aruco_params.maxMarkerPerimeterRate = 4.0  # Increase the max perimeter rate


# Start video capture
cap = cv2.VideoCapture(0)  # Use your camera index or video file path
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,3840)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,2160)
# Read camera properties
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps          = cap.get(cv2.CAP_PROP_FPS)  # sometimes returns 0 for webcams


fourcc = cv2.VideoWriter_fourcc(*'XVID')
#    - VideoWriter takes:
#        * output filename ("output.avi")
#        * fourcc code
#        * frames per second (e.g., 20.0)
#        * frame size (width, height) which must match the captured frame size
if(fps==0):fps=30
out = cv2.VideoWriter('AprilAndArucoDifferentDistances2.avi', fourcc,fps , (frame_width, frame_height))


def detect(dictionary,aruco_params,family,gray,frame,frameIndex):

    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=aruco_params)

    if ids is not None:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    



frameIndex=0
startTime= time.time()
print("Press 'q' to quit.")
while True:
    frameIndex+=1
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    out.write(frame)

        # Convert to grayscale
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #detect(aruco_dict,aruco_params,"ArUco",gray,frame,frameIndex)
    #detect(april_dict,aruco_params,"AprilTag",gray,frame,frameIndex)


    # Display the frame
    cv2.imshow("Detection", frame)
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

print(f"frame index:{frameIndex}")
detectionTime=time.time() - startTime
print(f" detectionTime:{detectionTime}")
print(f" fps:{frameIndex/detectionTime}")
cap.release()
cv2.destroyAllWindows()