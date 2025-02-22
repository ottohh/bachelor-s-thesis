import cv2
import numpy as np
from readCameraAndTagCoordinates import calculate_aruco_corners, calculate_april_corners, getCameraRotAndTrans
import pickle
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import os


# Define the dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
april_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
# Define the marker side length in meters
MARKER_LENGTH = 0.05  # For example, 5 cm



aruco_params = cv2.aruco.DetectorParameters()


aruco_params.adaptiveThreshWinSizeMin = 5
aruco_params.adaptiveThreshWinSizeMax = 50  # Increase max size for larger markers
aruco_params.adaptiveThreshWinSizeStep = 5

aruco_params.minMarkerPerimeterRate = 0.05 # Keyboard keys are detected as markers so had to increase this.
aruco_params.maxMarkerPerimeterRate = 4.0  # Increase the max perimeter rate
aruco_params.polygonalApproxAccuracyRate = 0.02


# Load camera calibration data
calibration_file = "camera_calibration_data.npz"  # Path to saved calibration data
data = np.load(calibration_file)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

cameraOffsetMatrix=None
with open('offset.pickle', 'rb') as f:
    cameraOffsetMatrix = pickle.load(f)

cameraOffsetMatrix[:3, 3]/=1000

# Start video capture
cap = cv2.VideoCapture("AprilAndArucoPart2.avi")  # Use your camera index or video file path

startFrame=1000
cap.set(cv2.CAP_PROP_POS_FRAMES,startFrame)


print("Press 'q' to quit.")


def printMinSideLength(corners,family):
    polygon=corners[0].reshape(-1, 2)
    side_lengths = []
    for i2 in range(len(polygon)):
        pt1 = polygon[i2]
        pt2 = polygon[(i2 + 1) % len(polygon)]  # Loop back to the start for the last segment
        # Calculate Euclidean distance

        length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        side_lengths.append(length)
    if((family+"minSide" in result)==False):result[family+"minSide"]=[]

    result[family+"minSide"].append(min(side_lengths))
    #print(f"{family} min side {min(side_lengths)}")

def projectTopLeftCorner(rvec,tvec,MARKER_CORNERS_3D):
    imgPoints = cv2.projectPoints(MARKER_CORNERS_3D[0],rvec,tvec, camera_matrix, dist_coeffs)
    projected_point = tuple(map(int, imgPoints[0].ravel()))

    cv2.circle(frame,projected_point,4,(0,255,222),thickness=2)

def drawRejected(rejected,frame):
      # Draw rejected markers in red
    if rejected:
        for corner in rejected:
            corner = corner.reshape((4, 2)).astype(int)
            cv2.polylines(frame, [corner], isClosed=True, color=(0, 0, 255), thickness=2)
def drawAxes(frame,corners):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
            # Draw the axes of the marker
            # estimatePoseSingleMarkers uses hardcoded 3D points as the marker corners
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], MARKER_LENGTH / 2)


def detect(dictionary,aruco_params,family,gray,frame,frameIndex):

    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=aruco_params)
    MARKER_CORNERS_3D=None
    if(family=="ArUco"):MARKER_CORNERS_3D=calculate_aruco_corners()[0:4]
    else:
        MARKER_CORNERS_3D=calculate_april_corners()[0:4]

    #drawRejected(rejected,frame)
    

    if ids is not None:
        if(len(ids)>1 or ids[0]!=3):
            print(f"false positive {family}")
            return [None,None]
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
   
        for i, marker_id in enumerate(ids.flatten()):

            


            
            
 
            success, rvec, tvec = cv2.solvePnP(MARKER_CORNERS_3D, corners[i], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if success == False:print(success)
            #projectTopLeftCorner(rvec,tvec,MARKER_CORNERS_3D)
            #projectTopLeftCorner(rvec,tvec,calculate_april_corners()[0:4])
            #drawAxes(frame,corners)
            printMinSideLength(corners,family)
            
            
            return (rvec,tvec)
    return [None,None]


def calculateDistance(rvec,tvec,family):

        # Convert rotation vector to rotation matrix
        aprilCenter = np.ones(4)
        
        if(family=="AprilTag"):aprilCenter[0:3]=calculate_april_corners()[4]
        else: aprilCenter[0:3]=calculate_aruco_corners()[4]
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        transformation_matrix = np.eye(4, dtype=np.float32)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = tvec.flatten()
        estimatedCameraFramePoint = transformation_matrix @ aprilCenter
        
        

        (rot,pos)=getCameraRotAndTrans(startFrame)
        r = R.from_quat(rot)
        mTc= np.eye(4, dtype=np.float32)
        mTc[:3, :3] = r.as_matrix()
        mTc[:3, 3] = pos.flatten()

        oTc=cameraOffsetMatrix

        
        cameraFramePoint=oTc @np.linalg.inv(mTc) @ aprilCenter
        return (estimatedCameraFramePoint,cameraFramePoint)      
        


def save_result(family,estimatedCameraFramePoint,cameraFramePoint):

    if((family+"cameraFramePoints" in result) == False):result[family+"cameraFramePoints"]=[]
    if((family+"estimatedCameraFramePoint" in result) == False):result[family+"estimatedCameraFramePoint"]=[]

    result[family+"cameraFramePoints"].append(cameraFramePoint)
    result[family+"estimatedCameraFramePoint"].append(estimatedCameraFramePoint)



result = {}


   

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    (rvec,tvec)=detect(aruco_dict,aruco_params,"ArUco",gray,frame,startFrame)
    (rvec2,tvec2)=detect(april_dict,aruco_params,"AprilTag",gray,frame,startFrame)


    
    pc1 = None
    pc2 = None
    
    if(rvec is not None):
        (estimatedCameraFramePoint,cameraFramePoint) =calculateDistance(rvec,tvec,"ArUco")
        save_result("ArUco",estimatedCameraFramePoint,cameraFramePoint)

    if(rvec2 is not None):
        (estimatedCameraFramePoint,cameraFramePoint) = calculateDistance(rvec2,tvec2,"AprilTag")
        save_result("AprilTag",estimatedCameraFramePoint,cameraFramePoint)

            

    startFrame+=1


    """     # Display the frame
    cv2.imshow("Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break """




# 4. Plot
plt.figure(figsize=(8,5))

rawData=[]

def showData(family):
    diff=[]
    for estimated, correct in zip(result[family+"estimatedCameraFramePoint"], result[family+"cameraFramePoints"]):
        diff.append(np.linalg.norm(estimated[0:3]-correct[0:3])*100)
    # Get the corresponding minSide lengths
    side_lengths = result[family + "minSide"]

    print(f"Family {family} min {np.min(diff)}")
    print(f"Family {family} max {np.max(diff)}")
    print(f"Family {family} mean {np.mean(diff)}")
    print(f"Family {family} std {np.std(diff)}")
    print(f"Family {family} count {len(diff)}")

    # Create a dictionary to hold diffs, keyed by pixel index.
    diff_by_pixel = defaultdict(list)

    # Bin each distance by its "rounded" pixel index:
    for d, s in zip(diff, side_lengths):
        # If you want pixel 16 to correspond to 15.5 <= s < 16.5, do:
        pixel_index = int(np.floor(s + 0.5))
        diff_by_pixel[pixel_index].append(d)
        rawData.append({"family":family,"error":d,"minSide":pixel_index})


    # Print out statistics for each pixel bin in ascending order:
    px_values   = []
    min_values  = []
    mean_values = []
    max_values  = []
    counts = []
    for px in sorted(diff_by_pixel.keys()):
        values = diff_by_pixel[px]
        std = np.std(values)
        mean = np.mean(values)
        min = np.min(values)
        max = np.max(values)
        px_values.append(px)
        mean_values.append(mean)
        min_values.append(min)
        max_values.append(max)
        count = len(values)
        counts.append(count)

    if family=="ArUco":
        plt.plot(px_values, min_values,  label=f'Pienin {family}',  color='blue')
        plt.plot(px_values, mean_values, label=f'keskiarvo {family}', color='green')
        plt.plot(px_values, max_values,  label=f'Suurin {family}',  color='red')
    else :
        plt.plot(px_values, min_values,  label=f'Pienin {family}',  color='gray')
        plt.plot(px_values, mean_values, label=f'keskiarvo {family}', color='olive')
        plt.plot(px_values, max_values,  label=f'Suurin {family}',  color='cyan')
    
showData("ArUco")
showData("AprilTag")

plt.xlabel("Lyhyimmän sivun pituus (px)")
plt.ylabel("Virhe(cm)")

plt.legend()
plt.grid(True)
plt.show()

cap.release()
cv2.destroyAllWindows()