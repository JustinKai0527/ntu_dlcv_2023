import cv2
import os

# Define the source directory and colormap
src_directory = 'depth_visualize'
colormap = cv2.COLORMAP_JET  # Choose the colormap you want to apply

# Create an output directory
output_directory = 'colored_images'
os.makedirs(output_directory, exist_ok=True)

# Loop through each file in the source directory
for filename in os.listdir(src_directory):
    # Construct the full file path
    file_path = os.path.join(src_directory, filename)
    
    # Check if the file is an image
    if file_path.lower().endswith(('.png')):
        # Read the image
        src_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Ensure the image is in grayscale

        # Apply the colormap
        colored_image = cv2.applyColorMap(src_image, colormap)
        
        # Save the colored image to the output directory
        output_file_path = os.path.join(output_directory, filename)
        cv2.imwrite(output_file_path, colored_image)
        
        # Optionally display the image
        # cv2.imshow('Colored Image', colored_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()