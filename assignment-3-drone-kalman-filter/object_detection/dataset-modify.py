import os


image_folder = "D:\Drone-Kalman-Filters\drone-kalman-filters\\2-obj-detection-model\\bikecycle-images"
input_folder = "D:\Drone-Kalman-Filters\drone-kalman-filters\\2-obj-detection-model\\bikecycle-labels"
output_folder = "D:\Drone-Kalman-Filters\drone-kalman-filters\\2-obj-detection-model\\bikecycle-labels-new"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over each file in the input folder
for filename in os.listdir(input_folder):
    inp_path = os.path.join(input_folder, filename)
    out_path = os.path.join(output_folder, filename)
    
    # Open input file for reading and output file for writing
    with open(inp_path, 'r') as infile, open(out_path, 'w') as outfile:
        # Iterate over each line in the input file
        for line in infile:
            # Check if the first character of the line is '0', '1', or '2'
            first_char = line.split()[0]
            if first_char in ['0','1']:
                # Replace '0' with '1', '1' with '2', and '2' with '3'
                first_char_new = str(int(first_char)+2)
                modified_line = first_char_new + line[len(first_char):]
                # Write the modified line to the output file
                outfile.write(modified_line)
    
    # Check if the output file is empty
    if os.path.getsize(out_path) == 0:
        os.remove(out_path)

# Get the base names of the text files in the output folder
text_basename = {os.path.splitext(file)[0] for file in os.listdir(output_folder)}

# Iterate over each image file in the image folder
for image_filename in os.listdir(image_folder):
    image_basename = os.path.splitext(image_filename)[0]
    
    # Check if the image base name matches any of the base names of text files
    if image_basename not in text_basename:
        # If not, delete the image file
        os.remove(os.path.join(image_folder, image_filename))