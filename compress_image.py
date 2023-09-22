import cv2

# Load the image
input_image_path = 'Nilesh_citizenship.jpg'
output_image_path = 'Nilesh_citizenship_compressed_image.jpg'

# Read the image
image = cv2.imread(input_image_path)

# Define the compression parameters (JPEG format)
compression_params = [cv2.IMWRITE_JPEG_QUALITY, 40]  # Adjust quality as needed (0-100)

# Compress and save the image
cv2.imwrite(output_image_path, image, compression_params)

print(f"Image compressed and saved to {output_image_path}")
