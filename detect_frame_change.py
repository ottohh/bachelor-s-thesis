import cv2
import numpy as np






# Start video capture
cap = cv2.VideoCapture("markerArucoAndApril.avi")  # Use your camera index or video file path

FPS=29.911815370238816

def mse(imageA, imageB):
    """
    Compute the Mean Squared Error between two images.
    The images must have the same dimensions.
    """
    # Convert images to float to avoid overflow or underflow
    diff = (imageA.astype("float") - imageB.astype("float")) ** 2
    # Sum of squared differences
    ssd = np.sum(diff)
    # Divide by the number of pixels
    mse_value = ssd / float(imageA.shape[0] * imageA.shape[1])
    return mse_value


frameIndex=0
prev_gray = None
print("Press 'q' to quit.")
while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break
   
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate MSE only if there is a previous frame to compare with
    if prev_gray is not None:
        current_mse = mse(gray, prev_gray)
        if(current_mse>1):print(F"MSE:{current_mse} and frame:{frameIndex}, seconds:{frameIndex/40}")
    frameIndex+=1
    # Update the previous frame
    prev_gray = gray

    # Display the frame
    cv2.imshow("Detection", frame)
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()