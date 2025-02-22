import os
import io
import zipfile
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive

# Dataset URLs
dataset_urls = [
    # "https://data.pix4d.com/misc/example_datasets/example_Island_Dominica_2017_Hurricane.zip",
    "https://data.pix4d.com/misc/example_datasets/Pix4Dmatic_example_1469_images.zip",
    # "https://data.pix4d.com/misc/example_datasets/example_belleview.zip"
]

# Step 1: Mount Google Drive for saving results
drive.mount('/content/drive')

# Google Drive output folder
cloud_path = "/content/drive/My Drive/segmented_images/"
os.makedirs(cloud_path, exist_ok=True)

for dataset_url in dataset_urls:
    dataset_name = os.path.splitext(os.path.basename(dataset_url))[0]
    dataset_extract_path = f"extracted_data/{dataset_name}"
    os.makedirs(dataset_extract_path, exist_ok=True)

    # Download and extract dataset
    response = requests.get(dataset_url, stream=True)
    if response.status_code == 200:
        print(f"Downloading {dataset_name} dataset... This may take a while.")
        with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
            zip_ref.extractall(dataset_extract_path)
        print(f"Dataset {dataset_name} extracted successfully!")
    else:
        print(f"Failed to download {dataset_name} dataset. Status code: {response.status_code}")
        continue

    # Debugging - List extracted files
    print(f"Extracted files in {dataset_name}:", os.listdir(dataset_extract_path))

    # Step 2: Find all images inside extracted folder, including subdirectories
    image_extensions = ('.jpg', '.png', '.jpeg', '.tif', '.tiff')
    image_files = []
    for root, _, files in os.walk(dataset_extract_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))

    # Step 3: Check if images were found
    if not image_files:
        print(f"No image files found in {dataset_name} dataset!")
        continue

    # Create dataset output folder in Google Drive
    dataset_folder = os.path.join(cloud_path, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)

    # Step 4: Process and save each image
    for img_path in image_files:
        print(f"Processing Image: {img_path}")
        
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert BGR to HSV

        # Define HSV color range for classification
        lower_vegetation = np.array([25, 40, 40])
        upper_vegetation = np.array([90, 255, 255])
        lower_water = np.array([90, 50, 50])
        upper_water = np.array([140, 255, 255])

        # Create masks
        vegetation_mask = cv2.inRange(image_hsv, lower_vegetation, upper_vegetation)
        water_mask = cv2.inRange(image_hsv, lower_water, upper_water)
        land_mask = cv2.bitwise_not(cv2.bitwise_or(vegetation_mask, water_mask))

        # Apply masks to segment images
        vegetation_segment = cv2.bitwise_and(image_rgb, image_rgb, mask=vegetation_mask)
        water_segment = cv2.bitwise_and(image_rgb, image_rgb, mask=water_mask)
        land_segment = cv2.bitwise_and(image_rgb, image_rgb, mask=land_mask)

        # Create subfolders for each image
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_folder = os.path.join(dataset_folder, img_name)
        os.makedirs(img_folder, exist_ok=True)

        # Save segmented images
        cv2.imwrite(os.path.join(img_folder, "vegetation.png"), vegetation_segment)
        cv2.imwrite(os.path.join(img_folder, "water.png"), water_segment)
        cv2.imwrite(os.path.join(img_folder, "land.png"), land_segment)

        print(f"Segmented images saved to {img_folder}")

print(f"All processed images saved to {cloud_path}")
