import os
from PIL import Image
import numpy as np

root_dir = 'assignment-2-video-search\\3-embedding-model\model-input-dataset'

# Define a list to store the image arrays
image_arrays = []

# Loop through each subfolder in the root directory
for imagePath in sorted(os.listdir(root_dir)):
    imageFolder = os.path.join(root_dir, imagePath)
    
    # Loop through each image file in the subfolder
    for filename in sorted(os.listdir(imageFolder)):
        # Construct the full path to the image file
        image_path = os.path.join(imageFolder, filename)
        
        # Open the image using PIL
        image = Image.open(image_path)
        
        # Resize the image to (128, 128)
        image = image.resize((128, 128))
        
        # Convert the image to grayscale
        image = image.convert('L')
        
        # Convert the image to a numpy array
        image_array = np.array(image)
        
        # Expand dimensions to (128, 128, 1)
        image_array = np.expand_dims(image_array, axis=-1)
        
        # Normalize the pixel values to the range [0, 1]
        image_array = image_array / 255.0
        
        # Append the image array to the list
        image_arrays.append(image_array)

# Convert the list of arrays to a numpy array
image_arrays = np.array(image_arrays)
print(image_arrays.shape)
# Save the padded images
np.savez("./mnistlikedataset224x1.npz", DataX=image_arrays)