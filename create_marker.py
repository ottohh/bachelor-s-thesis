import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

def saveMarkerToPdf(marker, pdfName, size_mm):
    """
    Save an ArUco marker to a PDF file with correct physical dimensions.

    Parameters:
        marker (numpy.ndarray): Marker image (grayscale or color).
        pdfName (str): Output PDF file name.
        size_mm (tuple): Marker size (width, height) in millimeters.
    """
    # Save marker as an image
    temp_image_file = "marker_temp.png"
    cv2.imwrite(temp_image_file, marker)

    # Convert size from mm to points
    mm_to_points = 72 / 25.4
    width_points = size_mm[0] * mm_to_points
    height_points = size_mm[1] * mm_to_points

    # Create a PDF canvas with A4 page size
    c = canvas.Canvas(pdfName, pagesize=A4)

    # Center the marker on the A4 page
    a4_width, a4_height = A4
    x_offset = (a4_width - width_points) / 2
    y_offset = (a4_height - height_points) / 2

    # Draw the marker at the correct size
    c.drawImage(temp_image_file, x_offset, y_offset, width=width_points, height=height_points)

        # Set font for the text you want to add
    c.setFont("Helvetica-Bold", 14)

    # Calculate a position above the marker
    # For example, 20 points above the marker's top edge, centered horizontally
    text_x = x_offset + (width_points / 2)
    text_y = y_offset + height_points + 20

    # Draw the PDF name (or any text) centered above the marker
    c.drawCentredString(text_x, text_y, pdfName)

    # Save the PDF
    c.save()

# Define the ArUco dictionary
#aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
MARKER_SIZE = 0.05
# Define the marker ID and size
MARKER_ID = 3  # Marker ID (must be in range for the dictionary, e.g., 0-99 for DICT_5X5_100)
PDFNAME = "markerApril"+str(MARKER_ID)+".pdf"


if __name__ == "__main__":

    dpi = 300
    marker_pixel_size = int((MARKER_SIZE*1000 / 25.4) * dpi)
    # Generate the marker
    marker = cv2.aruco.generateImageMarker(aruco_dict, MARKER_ID, marker_pixel_size)

    saveMarkerToPdf(marker,PDFNAME,(int(MARKER_SIZE*1000),int(MARKER_SIZE*1000)))



