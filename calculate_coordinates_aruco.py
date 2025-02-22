import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from math import pi, cos, sin
import time
import pickle


# Define the dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# Define the marker side length in meters
MARKER_LENGTH = 0.10  # For example, 5 cm
#Corner positions starting from top left.
MARKER_CORNERS_3D = np.array([
                            [-MARKER_LENGTH/2, MARKER_LENGTH/2, 0],     
                            [MARKER_LENGTH/2, MARKER_LENGTH/2, 0],     
                            [MARKER_LENGTH/2, -MARKER_LENGTH/2, 0], 
                            [-MARKER_LENGTH/2, -MARKER_LENGTH/2, 0]      
                        ])



aruco_params = cv2.aruco.DetectorParameters()


aruco_params.adaptiveThreshWinSizeMin = 5
aruco_params.adaptiveThreshWinSizeMax = 50  # Increase max size for larger markers
aruco_params.adaptiveThreshWinSizeStep = 5

#aruco_params.minMarkerPerimeterRate = 0.05 # Keyboard keys are detected as markers so had to increase this.
aruco_params.maxMarkerPerimeterRate = 4.0  # Increase the max perimeter rate



# Load camera calibration data
calibration_file = "camera_calibration_data.npz"  # Path to saved calibration data
data = np.load(calibration_file)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]



def draw_camera():

    glBegin(GL_QUADS)
    # Front face
    glColor3f(1, 0, 0)  # Red
    glVertex3f(-1, -0.5, 0.5)
    glVertex3f(1, -0.5, 0.5)
    glVertex3f(1, 0.5, 0.5)
    glVertex3f(-1, 0.5, 0.5)

    # Back face
    glColor3f(0, 1, 0)  # Green
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(-0.5, 0.5, -0.5)

    # Left face
    glColor3f(0, 0, 1)  # Blue
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(-0.5, -0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, -0.5)

    # Right face
    glColor3f(1, 1, 0)  # Yellow
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(0.5, 0.5, -0.5)
    glEnd()
    draw_semicircle(1, 30)


# Define a function to draw the semicircle
def draw_semicircle(radius, segments):
    glColor3f(1, 0, 0)
    glBegin(GL_QUAD_STRIP)

    for i in range(segments + 1):
        angle = pi * i / segments  # Angle from 0 to pi
        x = radius * cos(angle)   # Calculate x coordinate
        z = radius * sin(angle)   # Calculate z coordinate
        glVertex3f(x, 0.5, z + 0.5)  # Add the vertex upper to the semicircle
        glVertex3f(x, -0.5, z + 0.5)  # Add the vertex lower to the semicircle
    glEnd()
# Initialize Pygame and OpenGL

pygame.init()
screen = pygame.display.set_mode((800, 600), pygame.DOUBLEBUF | pygame.OPENGL)
gluPerspective(45, (800 / 600), 0.1, 50.0)
glEnable(GL_DEPTH_TEST)

glTranslatef(0, 0, -20)  # Move the camera back to view the object 

# Start video capture
cap = cv2.VideoCapture(0)  # Use your camera index or video file path


minPixelSize=10000000

frameIndex=0
startTime= time.time()
print("Press 'q' to quit.")
while True:
    frameIndex+=1
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is not None:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        
        # Estimate pose for each detected marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs,objPoints=MARKER_CORNERS_3D)
        for i, marker_id in enumerate(ids.flatten()):
            # Draw the axes of the marker
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], MARKER_LENGTH / 2)

            # Extract rotation and translation vectors
            rvec, tvec = rvecs[i], tvecs[i]


            # You can transform tvec to world coordinates if you know the camera's pose

            # Convert rotation vector to matrix and change rotation directions
            rvec[0][1]=rvec[0][1]*(-1)
            rvec[0][2]=rvec[0][2]*(-1)
            rvec[0][0]=rvec[0][0]*(-1)
            rotation_matrix, _ = cv2.Rodrigues(rvec)

        

            transformation_matrix = np.eye(4, dtype=np.float32)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = tvec.flatten()

            # Annotate the frame with the distance
            distance = np.linalg.norm(tvec)

            position = tuple(corners[i][0][0].astype(int))  # Top-left corner of the marker
            cv2.putText(frame, f"ID: {marker_id} Dist: {distance:.2f}m", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            # Apply transformations
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glPushMatrix()
            glMultMatrixf(transformation_matrix.T.flatten())  # Apply the transformation matrix

            draw_camera()
            glPopMatrix()
                # Update the display
            pygame.display.flip()



    # Display the frame
    cv2.imshow("Detection", frame)
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()