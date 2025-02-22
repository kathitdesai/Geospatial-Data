from google.colab import drive
import zipfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Mount Google Drive
drive.mount('/content/drive')

# Define paths (Upload ZIP to Google Drive first)
zip_path = "/content/drive/My Drive/example_Island_Dominica_2017_Hurricane.zip"  # Change this if needed
extract_path = "extracted_data"

# Step 2: Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Step 3: Debugging - List all extracted files
print("Extracted files and directories:", os.listdir(extract_path))

# Step 4: Find all images inside extracted folder, including subdirectories
image_extensions = ('.jpg', '.png', '.jpeg', '.tif', '.tiff')
image_files = []

for root, _, files in os.walk(extract_path):
    for file in files:
        if file.lower().endswith(image_extensions):
            image_files.append(os.path.join(root, file))

# Step 5: Check if images were found
if not image_files:
    raise ValueError("No image files found in the dataset!")

# Step 6: Load the first image for analysis
sample_img_path = image_files[0]
print(f"Processing Image: {sample_img_path}")

image = cv2.imread(sample_img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert BGR to HSV

# Step 7: Define HSV color range for classification
# Vegetation (Green Areas)
lower_vegetation = np.array([25, 40, 40])
upper_vegetation = np.array([90, 255, 255])

# Water (Blue Areas)
lower_water = np.array([90, 50, 50])
upper_water = np.array([140, 255, 255])

# Step 8: Create masks
vegetation_mask = cv2.inRange(image_hsv, lower_vegetation, upper_vegetation)
water_mask = cv2.inRange(image_hsv, lower_water, upper_water)

# Land (everything except vegetation & water)
land_mask = cv2.bitwise_not(cv2.bitwise_or(vegetation_mask, water_mask))

# Step 9: Apply masks to segment images
vegetation_segment = cv2.bitwise_and(image_rgb, image_rgb, mask=vegetation_mask)
water_segment = cv2.bitwise_and(image_rgb, image_rgb, mask=water_mask)
land_segment = cv2.bitwise_and(image_rgb, image_rgb, mask=land_mask)

# Step 10: Display results
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(vegetation_segment)
axes[1].set_title("Vegetation Detection")
axes[1].axis("off")

axes[2].imshow(water_segment)
axes[2].set_title("Water Detection")
axes[2].axis("off")

axes[3].imshow(land_segment)
axes[3].set_title("Land Detection")
axes[3].axis("off")

plt.show()

# Step 11: Save segmented images to Google Drive
cloud_path = "/content/drive/My Drive/segmented_images/"
os.makedirs(cloud_path, exist_ok=True)

cv2.imwrite(os.path.join(cloud_path, "vegetation.png"), vegetation_segment)
cv2.imwrite(os.path.join(cloud_path, "water.png"), water_segment)
cv2.imwrite(os.path.join(cloud_path, "land.png"), land_segment)

print(f"Images saved to {cloud_path}")
