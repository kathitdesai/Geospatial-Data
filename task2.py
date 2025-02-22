import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def optimized_count_plants(image_path):
    # Load the image
    image = io.imread(image_path)

    # Convert to HSV color space for better segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define green color range (adjust as needed)
    lower_green = np.array([25, 40, 40])  # Lower bound for green in HSV
    upper_green = np.array([90, 255, 255])  # Upper bound for green in HSV

    # Create a mask to isolate green plants
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply bitwise operation to extract green areas
    green_plants = cv2.bitwise_and(image, image, mask=mask)

    # Convert extracted green areas to grayscale
    gray = cv2.cvtColor(green_plants, cv2.COLOR_RGB2GRAY)

    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to remove noise
    min_contour_area = 30
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Draw contours on the original image
    output_image = image.copy()
    cv2.drawContours(output_image, valid_contours, -1, (0, 255, 0), 2)

    # Display the refined results
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Detected Plants: {len(valid_contours)}")
    plt.imshow(output_image)
    plt.axis("off")

    plt.show()

    return len(valid_contours)


# Example usage
image_path = r"C:\Users\kathit\Downloads\Count1.tif"  # Ensure correct file path format
plant_count = optimized_count_plants(image_path)
if plant_count is not None:
    print(f"Total plant count: {plant_count}")
