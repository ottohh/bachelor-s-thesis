import pandas as pd
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


df = pd.read_csv(
    "AprilAndArucoPart2.csv",
    skiprows=2,         # skip row indices 0 and 1 (the general info + empty line)
    header=[0, 1, 2, 3, 4],  
    # That means the next 5 lines we read (i.e. lines 2..6 in the file) 
    # become a 5-level MultiIndex for the columns
)

START_FRAME_FOR_MOCAP=0
START_FRAME_FOR_CAMERA=-3

#Mocap fps/video fps
FPS_difference=100/29.9118

MARKER_SIZE=0.05 #In meters
paperWidth=0.21 #In meters
paperHeight=0.297 #In meters
arucoCorners=None
def calculate_aruco_corners():
    global arucoCorners
    if arucoCorners is not None:
        return arucoCorners
    rowId=0
    row=df.iloc[rowId]
    c1=np.array([row[x] for x in columnsForMarkerPaperCorners["aruco:Marker"+"1"]])
    c2=np.array([row[x] for x in columnsForMarkerPaperCorners["aruco:Marker"+"2"]])
    c3=np.array([row[x] for x in columnsForMarkerPaperCorners["aruco:Marker"+"3"]])
        # --- Compute unit vectors that define the paper plane directions ---
    # Bottom edge direction (left -> right)
    bottom_left_to_right = c3 - c2
    bottom_left_to_right_unit = bottom_left_to_right / np.linalg.norm(bottom_left_to_right)
    
    # Left edge direction (bottom -> top)
    bottom_to_top = c1 - c2
    bottom_to_top_unit = bottom_to_top / np.linalg.norm(bottom_to_top)
    
    # --- Compute the center of the paper (where marker center should be) ---
    # Start at bottom-left corner (c2) + half the paper width + half the paper height
    center_of_paper = ( c2
                        + bottom_left_to_right_unit * (paperWidth  / 2.0)
                        + bottom_to_top_unit         * (paperHeight / 2.0) )
    
    # --- Marker is in the center, so its center is the same as center_of_paper ---
    center_of_marker = center_of_paper
    
    # Half of marker side length
    half_marker = MARKER_SIZE / 2.0
    
    # --- Compute each corner of the marker ---
    # We define +x along bottom_left_to_right_unit, +y along bottom_to_top_unit
    # top-left     = center - x_half + y_half
    # top-right    = center + x_half + y_half
    # bottom-right = center + x_half - y_half
    # bottom-left  = center - x_half - y_half

    marker_top_left = ( center_of_marker 
                        - half_marker * bottom_left_to_right_unit
                        + half_marker * bottom_to_top_unit )
    
    marker_top_right = ( center_of_marker
                         + half_marker * bottom_left_to_right_unit
                         + half_marker * bottom_to_top_unit )
    
    marker_bottom_right = ( center_of_marker
                            + half_marker * bottom_left_to_right_unit
                            - half_marker * bottom_to_top_unit )
    
    marker_bottom_left = ( center_of_marker
                           - half_marker * bottom_left_to_right_unit
                           - half_marker * bottom_to_top_unit )
    return np.array([
                            marker_top_left,     
                            marker_top_right,     
                            marker_bottom_right, 
                            marker_bottom_left,
                            center_of_paper
                        ])
aprilCorners=None
def calculate_april_corners():
    global aprilCorners
    if(aprilCorners is not None):return aprilCorners
    rowId=0
    row=df.iloc[rowId]
    c1 = np.array([row[x] for x in columnsForMarkerPaperCorners["aruco:Marker"+"1"]])
    c2 = np.array([row[x] for x in columnsForMarkerPaperCorners["aruco:Marker"+"2"]])
    c3 = np.array([row[x] for x in columnsForMarkerPaperCorners["aruco:Marker"+"3"]])
        # --- Compute unit vectors that define the paper plane directions ---
    # Bottom edge direction (left -> right)
    left_to_right = c2 - c1
    left_to_right_unit = left_to_right / np.linalg.norm(left_to_right)
    
    # Left edge direction (bottom -> top)
    bottom_to_top = c2 - c3
    bottom_to_top_unit = bottom_to_top / np.linalg.norm(bottom_to_top)
    
    # --- Compute the center of the paper (where marker center should be) ---
    # Start at top-left corner (c2) + half the paper width - half the paper height
    center_of_paper = ( c2
                        + left_to_right_unit * (paperWidth  / 2.0)
                        - bottom_to_top_unit         * (paperHeight / 2.0) )
    
    # --- Marker is in the center, so its center is the same as center_of_paper ---
    center_of_marker = center_of_paper
    
    # Half of marker side length
    half_marker = MARKER_SIZE / 2.0
    
    # --- Compute each corner of the marker ---
    # We define +x along bottom_left_to_right_unit, +y along bottom_to_top_unit
    # top-left     = center - x_half + y_half
    # top-right    = center + x_half + y_half
    # bottom-right = center + x_half - y_half
    # bottom-left  = center - x_half - y_half

    marker_top_left = ( center_of_marker 
                        - half_marker * left_to_right_unit
                        + half_marker * bottom_to_top_unit )
    
    marker_top_right = ( center_of_marker
                         + half_marker * left_to_right_unit
                         + half_marker * bottom_to_top_unit )
    
    marker_bottom_right = ( center_of_marker
                            + half_marker * left_to_right_unit
                            - half_marker * bottom_to_top_unit )
    
    marker_bottom_left = ( center_of_marker
                           - half_marker * left_to_right_unit
                           - half_marker * bottom_to_top_unit )

    return np.array([
                            marker_top_left,     
                            marker_top_right,     
                            marker_bottom_right, 
                            marker_bottom_left,
                            center_of_paper
                        ])

def getCameraRotAndTrans(frame):
    rowId=int(START_FRAME_FOR_MOCAP+(frame-START_FRAME_FOR_CAMERA)*FPS_difference)
    pos = df.iloc[rowId, columnsForMarkerPaperCorners["WebCamPosition"]].values[0:3]
    rot = df.iloc[rowId, columnsForMarkerPaperCorners["WebCamRotation"]].values[0:4]
    return (rot,pos)


columnNameForApril="april:Marker"
columnNameForAruco="aruco:Marker"
columnNameForWebCam="WebCam"
columnsForMarkerPaperCorners = {"WebCamRotation":[],"WebCamPosition":[]}
def createDicts(all_cols):
    index=0
    for col in all_cols:
        if(col[0]=="Marker" and (col[1].startswith(columnNameForApril)or col[1].startswith(columnNameForAruco) )and col[3]=="Position"):
            if((col[1] in columnsForMarkerPaperCorners)==False):columnsForMarkerPaperCorners[col[1]]=[]

            columnsForMarkerPaperCorners[col[1]].append(index)

        if col[1]==columnNameForWebCam:
            if(col[3]=="Position"):
                columnsForMarkerPaperCorners["WebCamPosition"].append(index)
            if(col[3]=="Rotation"): 
                columnsForMarkerPaperCorners["WebCamRotation"].append(index)

        index+=1

createDicts(df.columns.tolist())




#This is used to find the correct starting frame to sync the mocap output with video frames
# No need to touch it
def main():
    # Extract the first measurement for Position and Rotation
    first_pos = df.iloc[0, columnsForMarkerPaperCorners["WebCamPosition"]].values
    first_rot = df.iloc[0, columnsForMarkerPaperCorners["WebCamRotation"]].values

    # Example threshold. You can adjust based on your needs.
    POSITION_MSE_THRESHOLD = 0.00
    ROTATION_MSE_THRESHOLD = 0.00
    # Loop over all subsequent rows
    for i in range(1, len(df)):
        current_pos = df.iloc[i, columnsForMarkerPaperCorners["WebCamPosition"]].values
        current_rot = df.iloc[i, columnsForMarkerPaperCorners["WebCamRotation"]].values
        if(i>1000):break
        # Compute MSE for position
        mse_pos = np.mean((first_pos - current_pos) ** 2)
        if mse_pos > POSITION_MSE_THRESHOLD:
            print(f"Row {i}: Position MSE {mse_pos:.4f} exceeds threshold on second {i/100}")
            continue
        
        # Compute MSE for rotation
        mse_rot = np.mean((first_rot - current_rot) ** 2)
        if mse_rot > ROTATION_MSE_THRESHOLD:
            print(f"Row {i}: Rotation MSE {mse_rot:.4f} exceeds threshold on second {i/100}")




