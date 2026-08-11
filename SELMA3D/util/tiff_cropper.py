import os
from skimage import io, transform
from skimage.util import img_as_ubyte

# Define input and output directories
input_dir = '../data/microscopy_image/microscopy_image/c-Fos_brain_cells/sample1/C01'
output_dir = '../data/microscopy_image/microscopy_image/c-Fos_brain_cells/sample1/C01_cropped'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to resize image while maintaining aspect ratio
def resize_image(image, max_size):
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        image = transform.resize(image, (new_height, new_width), anti_aliasing=True)
    return img_as_ubyte(image)

# Iterate through .tif files in the input directory
for file in os.listdir(input_dir):
    if file.endswith('.tif'):
        file_path = os.path.join(input_dir, file)
        try:
            image = io.imread(file_path)
            resized_image = resize_image(image, 2048)
            output_path = os.path.join(output_dir, file)
            io.imsave(output_path, resized_image)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

print("Cropping and resizing completed.")
